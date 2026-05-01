// orion_gpu_ntt.h — GPU-accelerated NTT via Metal
//
// Implements negacyclic NTT over Z_q[x]/(x^n+1) on Metal GPU.
// Used for ring arithmetic in lattice commitments and folding.
//
// Ring parameters are configurable. Default uses Dilithium-like:
//   q = 8383489, n = 256, w = 1753 (2^23 root of unity)
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework Metal -ldl \
//     -I . -I core \
//     core/orion_gpu_ntt.m -c
//
// Run:
//   (use via test_gpu_ntt.m)

#ifndef ORION_GPU_NTT_H
#define ORION_GPU_NTT_H

#import <stdbool.h>
#import <stdint.h>

// ============================================================================
// Ring Parameters
// ============================================================================

/// Default modulus (Dilithium-3 q)
#define GPU_NTT_Q 8383489

/// Polynomial degree (must be power of 2 for standard NTT)
#define GPU_NTT_N 256

/// Primitive root for modulus (w^((q-1)/2n) = -1)
#define GPU_NTT_W 1753

// ============================================================================
// Ring Element
// ============================================================================

/// A single polynomial in coefficient form (size = n)
typedef struct {
    uint32_t coeff[GPU_NTT_N];
} GPUNTTPoly;

/// A single ring element (also n coefficients)
typedef GPUNTTPoly GPUNTTElement;

// ============================================================================
// GPU Context
// ============================================================================

/// Opaque GPU context handle
typedef struct OrionGPUContext O_RIONGPUContext;

/// Initialize Metal GPU context
/// @return new context or NULL on failure
O_RIONGPUContext *orion_gpu_init(void);

/// Release GPU context and resources
void orion_gpu_release(O_RIONGPUContext *ctx);

// ============================================================================
// NTT Operations
// ============================================================================

/// Forward NTT: coefficient → NTT domain
/// @param ctx GPU context
/// @param input Input polynomial (coeff form)
/// @param output Output polynomial (NTT form, pre-allocated)
/// @return true on success
bool orion_ntt_forward(
    O_RIONGPUContext *ctx,
    const GPUNTTPoly *input,
    GPUNTTPoly *output
);

/// Inverse NTT: NTT domain → coefficient form
/// @param ctx GPU context
/// @param input Input polynomial (NTT form)
/// @param output Output polynomial (coeff form, pre-allocated)
/// @return true on success
bool orion_ntt_inverse(
    O_RIONGPUContext *ctx,
    const GPUNTTPoly *input,
    GPUNTTPoly *output
);

/// Pointwise multiplication in NTT domain
/// @param ctx GPU context
/// @param a First operand (NTT form)
/// @param b Second operand (NTT form)
/// @param result Product a * b mod q (NTT form, pre-allocated)
/// @return true on success
bool orion_ntt_multiply(
    O_RIONGPUContext *ctx,
    const GPUNTTPoly *a,
    const GPUNTTPoly *b,
    GPUNTTPoly *result
);

/// Compute NTT of multiple polynomials in batch
/// @param ctx GPU context
/// @param inputs Array of input polynomials
/// @param outputs Array of output polynomials (pre-allocated)
/// @param count Number of polynomials
/// @return true on success
bool orion_ntt_forward_batch(
    O_RIONGPUContext *ctx,
    const GPUNTTPoly *inputs,
    GPUNTTPoly *outputs,
    int count
);

// ============================================================================
// Ring Arithmetic
// ============================================================================

/// Add two polynomials: c = a + b mod q
void orion_poly_add(GPUNTTPoly *c, const GPUNTTPoly *a, const GPUNTTPoly *b);

/// Subtract two polynomials: c = a - b mod q
void orion_poly_sub(GPUNTTPoly *c, const GPUNTTPoly *a, const GPUNTTPoly *b);

/// Scalar multiplication: c = k * a mod q
void orion_poly_scalar_mul(GPUNTTPoly *c, uint32_t k, const GPUNTTPoly *a);

// ============================================================================
// Utility
// ============================================================================

/// Get last Metal device name for info
const char *orion_gpu_device_name(O_RIONGPUContext *ctx);

/// Check if GPU is available
bool orion_gpu_available(void);

#endif // ORION_GPU_NTT_H