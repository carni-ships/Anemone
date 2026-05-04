// orion_gpu_ntt.m — GPU-accelerated NTT via Metal
//
// Ported from zkMetal's lattice_ntt.metal for use in Anemone.
// Uses Apple's Metal GPU for position-dependent NTT butterfly operations.

#import "orion_gpu_ntt.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// ============================================================================
// Metal Kernel Source (ported from zkMetal lattice_ntt.metal)
// ============================================================================

// Dilithium field arithmetic (q = 8380417)
static const uint32_t DIL_Q = 8380417;

// Forward NTT kernel for Dilithium
static const char* DILITHIUM_NTT_KERNEL = R"(
#include <metal_stdlib>
using namespace metal;

// Dilithium field: q = 8380417
constant uint DIL_Q = 8380417U;

inline uint dil_reduce(ulong a) {
    return uint(a % ulong(DIL_Q));
}

inline uint dil_add(uint a, uint b) {
    ulong s = ulong(a) + ulong(b);
    return s >= ulong(DIL_Q) ? uint(s - ulong(DIL_Q)) : uint(s);
}

inline uint dil_sub(uint a, uint b) {
    return a >= b ? (a - b) : (a + DIL_Q - b);
}

inline uint dil_mul(uint a, uint b) {
    return dil_reduce(ulong(a) * ulong(b));
}

kernel void dilithium_ntt_batch(
    device uint* polys [[buffer(0)]],
    constant uint* twiddles [[buffer(1)]],
    constant uint& num_polys [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_polys) return;

    threadgroup uint shared_poly[256];

    uint base = tgid * 256;
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 1;
    for (uint len = 128; len >= 2; len >>= 1) {
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            uint tw = twiddles[k + block_idx];
            uint i0 = start + j;
            uint i1 = i0 + len;
            uint t = dil_mul(tw, shared_poly[i1]);
            uint u = shared_poly[i0];
            shared_poly[i0] = dil_add(u, t);
            shared_poly[i1] = dil_sub(u, t);
        }
        k += num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}
)";

// Inverse NTT kernel for Dilithium
static const char* DILITHIUM_INTT_KERNEL = R"(
#include <metal_stdlib>
using namespace metal;

constant uint DIL_Q = 8380417U;

inline uint dil_reduce(ulong a) {
    return uint(a % ulong(DIL_Q));
}

inline uint dil_add(uint a, uint b) {
    ulong s = ulong(a) + ulong(b);
    return s >= ulong(DIL_Q) ? uint(s - ulong(DIL_Q)) : uint(s);
}

inline uint dil_sub(uint a, uint b) {
    return a >= b ? (a - b) : (a + DIL_Q - b);
}

inline uint dil_mul(uint a, uint b) {
    return dil_reduce(ulong(a) * ulong(b));
}

kernel void dilithium_intt_batch(
    device uint* polys [[buffer(0)]],
    constant uint* fwd_twiddles [[buffer(1)]],
    constant uint& num_polys [[buffer(2)]],
    constant uint& inv_n [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_polys) return;

    threadgroup uint shared_poly[256];

    uint base = tgid * 256;
    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 127;
    for (uint len = 2; len <= 128; len <<= 1) {
        uint num_blocks = 256 / (2 * len);
        for (uint block = lid; block < num_blocks * len; block += tg_size) {
            uint block_idx = block / len;
            uint j = block % len;
            uint start = block_idx * 2 * len;
            uint fwd_tw = fwd_twiddles[k - block_idx];
            uint tw = (fwd_tw == 0) ? 0 : (DIL_Q - fwd_tw);
            uint i0 = start + j;
            uint i1 = i0 + len;
            uint t = shared_poly[i0];
            shared_poly[i0] = dil_add(t, shared_poly[i1]);
            shared_poly[i1] = dil_mul(tw, dil_sub(t, shared_poly[i1]));
        }
        k -= num_blocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        shared_poly[i] = dil_mul(shared_poly[i], inv_n);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared_poly[i];
    }
}
)";

// ============================================================================
// Context
// ============================================================================

struct OrionGpuNtt {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;

    // Dilithium kernels
    id<MTLComputePipelineState> dilithiumNtt;
    id<MTLComputePipelineState> dilithiumIntt;

    // Twiddle buffers
    id<MTLBuffer> twiddleBuffer;
    id<MTLBuffer> invTwiddleBuffer;
    id<MTLBuffer> invNBuffer;

    // Scratch buffer for in-place operations
    id<MTLBuffer> scratchBuffer;

    uint32_t maxPolys;
    uint32_t logN;
    uint32_t N;
    uint32_t q;
};

// ============================================================================
// Helper Functions
// ============================================================================

static uint32_t powmod(uint32_t base, uint32_t exp, uint32_t mod) {
    uint64_t result = 1;
    uint64_t b = base;
    while (exp) {
        if (exp & 1) result = (result * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (uint32_t)result;
}

// Compute primitive root for NTT of size N
static uint32_t compute_root(uint32_t N, uint32_t q) {
    // For Dilithium: g = 1753 is a primitive root of q
    // We need root of unity w = g^((q-1)/N) for forward NTT
    uint32_t phi = q - 1;
    uint32_t exp = phi / N;
    return powmod(1753, exp, q);
}

// ============================================================================
// API Implementation
// ============================================================================

OrionGpuNtt* orion_gpu_ntt_create(void) {
    if (!orion_gpu_ntt_available()) {
        return NULL;
    }

    OrionGpuNtt *ntt = (OrionGpuNtt *)calloc(1, sizeof(OrionGpuNtt));
    if (!ntt) return NULL;

    ntt->device = MTLCreateSystemDefaultDevice();
    ntt->commandQueue = [ntt->device newCommandQueue];

    NSError *error = nil;

    // Try to compile NTT kernel first
    NSError *nttError = nil;
    ntt->library = [ntt->device newLibraryWithSource:
        [NSString stringWithUTF8String:DILITHIUM_NTT_KERNEL]
        options:nil error:&nttError];
    if (!ntt->library) {
        NSLog(@"Failed to compile Dilithium NTT library: %@", nttError);
        free(ntt);
        return NULL;
    }

    // Log available functions
    NSArray *funcs = ntt->library.functionNames;
    NSLog(@"Available functions in library: %@", funcs);

    id<MTLFunction> nttFunc = [ntt->library newFunctionWithName:@"dilithium_ntt_batch"];
    if (!nttFunc) {
        NSLog(@"Failed to find ntt function");
        free(ntt);
        return NULL;
    }

    // Now compile INTT kernel separately
    NSError *inttError = nil;
    id<MTLLibrary> inttLib = [ntt->device newLibraryWithSource:
        [NSString stringWithUTF8String:DILITHIUM_INTT_KERNEL]
        options:nil error:&inttError];
    if (!inttLib) {
        NSLog(@"Failed to compile Dilithium INTT library: %@", inttError);
        free(ntt);
        return NULL;
    }

    id<MTLFunction> inttFunc = [inttLib newFunctionWithName:@"dilithium_intt_batch"];

    if (!inttFunc) {
        NSLog(@"Failed to find INTT function");
        free(ntt);
        return NULL;
    }

    NSError *err = nil;
    ntt->dilithiumNtt = [ntt->device newComputePipelineStateWithFunction:nttFunc error:&err];
    if (!ntt->dilithiumNtt) {
        NSLog(@"Failed to create Dilithium NTT pipeline: %@", err);
        free(ntt);
        return NULL;
    }

    ntt->dilithiumIntt = [ntt->device newComputePipelineStateWithFunction:inttFunc error:&err];
    if (!ntt->dilithiumIntt) {
        NSLog(@"Failed to create Dilithium INTT pipeline: %@", err);
        free(ntt);
        return NULL;
    }

    ntt->maxPolys = 0;
    ntt->logN = 8;  // 256 elements
    ntt->N = 256;
    ntt->q = DIL_Q;  // 8380417

    return ntt;
}

void orion_gpu_ntt_destroy(OrionGpuNtt *ntt) {
    if (!ntt) return;
    // Release Metal objects
    ntt->scratchBuffer = nil;
    ntt->invNBuffer = nil;
    ntt->invTwiddleBuffer = nil;
    ntt->twiddleBuffer = nil;
    ntt->dilithiumIntt = nil;
    ntt->dilithiumNtt = nil;
    ntt->library = nil;
    ntt->commandQueue = nil;
    ntt->device = nil;
    free(ntt);
}

bool orion_gpu_ntt_available(void) {
    return MTLCreateSystemDefaultDevice() != nil;
}

bool orion_gpu_ntt_init_dilithium(OrionGpuNtt *ntt, uint32_t max_polys) {
    if (!ntt || max_polys == 0) return false;

    // Allocate twiddle buffers
    uint32_t nTwiddles = ntt->N / 2;  // 128 twiddles for N=256

    ntt->twiddleBuffer = [ntt->device newBufferWithLength:
        nTwiddles * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    ntt->invTwiddleBuffer = [ntt->device newBufferWithLength:
        nTwiddles * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    ntt->invNBuffer = [ntt->device newBufferWithLength:
        sizeof(uint32_t) options:MTLResourceStorageModeShared];

    if (!ntt->twiddleBuffer || !ntt->invTwiddleBuffer || !ntt->invNBuffer) {
        return false;
    }

    // Generate forward twiddles
    uint32_t root = compute_root(ntt->N, ntt->q);
    uint32_t *tw = (uint32_t *)ntt->twiddleBuffer.contents;
    uint32_t w = 1;
    for (uint32_t i = 0; i < nTwiddles; i++) {
        tw[i] = w;
        w = (uint64_t)w * root % ntt->q;
    }

    // Generate reversed twiddles for inverse
    uint32_t *invTw = (uint32_t *)ntt->invTwiddleBuffer.contents;
    uint32_t w_inv = powmod(root, ntt->q - 2, ntt->q);  // root^{-1}
    for (uint32_t i = 0; i < nTwiddles; i++) {
        invTw[i] = powmod(w_inv, i, ntt->q);
    }

    // Compute inv_n = N^{-1} mod q
    uint32_t invN = powmod(ntt->N, ntt->q - 2, ntt->q);
    *(uint32_t *)ntt->invNBuffer.contents = invN;

    ntt->maxPolys = max_polys;

    // Allocate scratch buffer for in-place operations
    ntt->scratchBuffer = [ntt->device newBufferWithLength:
        ntt->N * max_polys * sizeof(uint32_t) options:MTLResourceStorageModeShared];

    return true;
}

bool orion_gpu_ntt_init_kyber(OrionGpuNtt *ntt, uint32_t max_polys) {
    // Kyber uses q=3329, different from Dilithium
    // For now, only Dilithium is implemented
    NSLog(@"Kyber NTT not yet implemented - use Dilithium");
    return false;
}

bool orion_gpu_ntt_forward_dilithium(OrionGpuNtt *ntt, uint32_t *data, uint32_t num_polys) {
    if (!ntt || !data || num_polys == 0 || num_polys > ntt->maxPolys) {
        return false;
    }

    // Create input buffer from data
    id<MTLBuffer> inputBuffer = [ntt->device newBufferWithBytes:data
        length:num_polys * ntt->N * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = ntt->commandQueue;
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:ntt->dilithiumNtt];
    [enc setBuffer:inputBuffer offset:0 atIndex:0];
    [enc setBuffer:ntt->twiddleBuffer offset:0 atIndex:1];

    uint32_t numPolys = num_polys;
    [enc setBytes:&numPolys length:sizeof(uint32_t) atIndex:2];

    // Calculate thread dispatch
    // 256 elements per polynomial, 8 elements per thread, 32 threads per TG
    // But we want at least 256 threads per TG for occupancy
    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);
    MTLSize threadgroups = MTLSizeMake(num_polys, 1, 1);

    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Copy result back
    memcpy(data, inputBuffer.contents, num_polys * ntt->N * sizeof(uint32_t));

    return true;
}

bool orion_gpu_ntt_inverse_dilithium(OrionGpuNtt *ntt, uint32_t *data, uint32_t num_polys) {
    if (!ntt || !data || num_polys == 0 || num_polys > ntt->maxPolys) {
        return false;
    }

    id<MTLBuffer> inputBuffer = [ntt->device newBufferWithBytes:data
        length:num_polys * ntt->N * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = ntt->commandQueue;
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:ntt->dilithiumIntt];
    [enc setBuffer:inputBuffer offset:0 atIndex:0];
    [enc setBuffer:ntt->invTwiddleBuffer offset:0 atIndex:1];

    uint32_t numPolys = num_polys;
    [enc setBytes:&numPolys length:sizeof(uint32_t) atIndex:2];

    uint32_t invN = *(uint32_t *)ntt->invNBuffer.contents;
    [enc setBytes:&invN length:sizeof(uint32_t) atIndex:3];

    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);
    MTLSize threadgroups = MTLSizeMake(num_polys, 1, 1);

    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    memcpy(data, inputBuffer.contents, num_polys * ntt->N * sizeof(uint32_t));

    return true;
}

bool orion_gpu_ntt_forward_kyber(OrionGpuNtt *ntt, uint16_t *data, uint32_t num_polys) {
    // Not yet implemented
    return false;
}

bool orion_gpu_ntt_inverse_kyber(OrionGpuNtt *ntt, uint16_t *data, uint32_t num_polys) {
    // Not yet implemented
    return false;
}

bool orion_gpu_ntt_roundtrip_dilithium(OrionGpuNtt *ntt, uint32_t *data, uint32_t num_polys) {
    if (!orion_gpu_ntt_forward_dilithium(ntt, data, num_polys)) {
        return false;
    }
    return orion_gpu_ntt_inverse_dilithium(ntt, data, num_polys);
}