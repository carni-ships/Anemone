// test_ntt_ane.m — NTT on ANE Feasibility Research
// Research question: Can NTT be expressed as ANE convolution chains?
//
// Key insight: NTT butterfly = add/sub + twiddle multiply
// - Add/sub: expressible as conv1x1 with [[1,1],[1,-1]] kernel
// - Twiddle multiply: element-wise scaling (can't be absorbed into weight)
//
// Hypothesis: For fixed-size NTT, twiddle factors can be baked into weights
// if we restructure as a series of diagonal matrix multiplications.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_ntt_ane.m -o test_ntt_ane
// Run:
//   ./test_ntt_ane

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "orion_mil_cache.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// NTT Constants (BabyBear, p = 2^32 - 2^25 + 1)
// Primitive root for N = 256: g = 15 (generator)
// ---------------------------------------------------------------------------
#define NTT_SIZE 256
#define NTT_LOG 8

// BabyBear field parameters
static const uint32_t kBabyBearP = 0x78000001;
static const uint32_t kBabyBearG = 15;  // primitive root

// ---------------------------------------------------------------------------
// Field arithmetic (CPU baseline)
// ---------------------------------------------------------------------------
static uint32_t babybear_add(uint32_t a, uint32_t b) {
    uint32_t c = a + b;
    if (c >= kBabyBearP) c -= kBabyBearP;
    return c;
}

static uint32_t babybear_sub(uint32_t a, uint32_t b) {
    return a < b ? a + kBabyBearP - b : a - b;
}

static uint32_t babybear_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    uint32_t t = (uint32_t)prod;
    uint32_t carry = (uint32_t)(prod >> 32);
    // Montgomery reduction for BabyBear
    // p = 2^32 - 2^25 + 1, so we can do fast reduction
    uint32_t c = carry << 25;
    c += carry >> 7;
    c += t;
    if (c >= kBabyBearP) c -= kBabyBearP;
    return c;
}

// ---------------------------------------------------------------------------
// CPU NTT (Cooley-Tukey radix-2)
// ---------------------------------------------------------------------------
static void cpu_ntt_inplace(uint32_t *a, int n, int log_n) {
    // Bit reversal permutation
    for (int i = 0; i < n; i++) {
        int j = 0;
        int x = i;
        for (int k = 0; k < log_n; k++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            uint32_t tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    // Butterfly stages
    for (int s = 1; s <= log_n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;

        // Twiddle factor for this stage
        // w = g^((N/m))
        uint32_t w = 1;
        uint32_t w_step = babybear_mul(kBabyBearG, kBabyBearG);  // g^2 for step
        for (int i = 0; i < s - 1; i++) w = babybear_mul(w, w_step);

        for (int i = 0; i < n; i += m) {
            uint32_t u = w;
            for (int j = 0; j < m2; j++) {
                uint32_t t = babybear_mul(a[i + j + m2], u);
                uint32_t a_ij = a[i + j];
                a[i + j] = babybear_add(a_ij, t);
                a[i + j + m2] = babybear_sub(a_ij, t);
                u = babybear_mul(u, w);
            }
            w = babybear_mul(w, w_step);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU NTT butterfly as convolution (for ANE comparison)
// ---------------------------------------------------------------------------
// Butterfly matrix (for 2 elements):
// [a']   [1  1] [a]
// [b'] = [1 -1] [b]
//
// This is a 2x2 conv with kernel [[1,1],[1,-1]]
// But ANE Conv1x1 is a single output channel operation...
//
// Actually, for ANE to compute the butterfly:
// - Channel 0: a' = a + b  (conv with kernel [1,1])
// - Channel 1: b' = a - b  (conv with kernel [1,-1])
//
// For dim=2 with seq=1, this is a 2x2 weight matrix.

static NSData *make_blob_butterfly(int dim) {
    int ws = dim * dim * 2; // fp16
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);

    // Butterfly kernel: [[1, 1], [1, -1]]
    // For identity mapping (no twiddle)
    fp16[0 * dim + 0] = (_Float16)1.0f;  // a' = a + b
    fp16[0 * dim + 1] = (_Float16)1.0f;
    fp16[1 * dim + 0] = (_Float16)1.0f;  // b' = a - b
    fp16[1 * dim + 1] = (_Float16)-1.0f;

    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Twiddle matrix: [[w, 0], [0, 1]] (multiply only first element by w)
// This is diagonal scaling, which ANE can't do as conv1x1 (it's pointwise)

// ---------------------------------------------------------------------------
// Build NTT butterfly MIL program
// dim = N (transform size), but ANE constraint: dim must be power of 2, >= 16
// ---------------------------------------------------------------------------
static NSString *build_butterfly_mil(int dim, int seq) {
    NSString *wpath = @"@model_path/weights/butterfly.bin";

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", dim, seq];
    [body appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%s\"), offset = uint64(64)))];\n", dim, dim, dim, dim, [wpath UTF8String]];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> c_out = conv(W, x16, stride = [1, 1], pad = valid, dilation = [1, 1], groups = 1)[name = string(\"bfly\")];\n", dim, seq];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = c_out)[name = string(\"out\")];\n", dim, seq];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// ---------------------------------------------------------------------------
// IOSurface helpers
// ---------------------------------------------------------------------------
static IOSurfaceRef make_fp32_surface(int channels, int seq_len) {
    int count = channels * seq_len;
    size_t bytes = count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    if (!s) return NULL;
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = (float)(i + 1);
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static void write_fp32_elem(IOSurfaceRef s, int idx, float val) {
    IOSurfaceLock(s, 0, NULL);
    ((float *)IOSurfaceGetBaseAddress(s))[idx] = val;
    IOSurfaceUnlock(s, 0, NULL);
}

static float read_fp32_elem(IOSurfaceRef s, int idx) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[idx];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// ---------------------------------------------------------------------------
// Test 1: Single butterfly (dim=2, the basic building block)
// ---------------------------------------------------------------------------
static void test_single_butterfly(void) {
    printf("\n=== Test: Single Butterfly (dim=2) ===\n");
    printf("  Butterfly: a' = a + b, b' = a - b\n");
    printf("  ANE can compute this via conv1x1 with [[1,1],[1,-1]] weights\n");

    int dim = 2;
    int seq = 1;

    NSString *prog_text = build_butterfly_mil(dim, seq);
    NSData *blob = make_blob_butterfly(dim);
    NSDictionary *wdict = @{@"@model_path/weights/butterfly.bin": @{@"offset": @0, @"data": blob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "butterfly");
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    // Input: a=3, b=5
    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_elem(ioX, 0, 3.0f);
    write_fp32_elem(ioX, 1, 5.0f);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        float a_prime = read_fp32_elem(ioY, 0);
        float b_prime = read_fp32_elem(ioY, 1);
        printf("  Input:  a=3, b=5\n");
        printf("  CPU:    a'=8, b'=-2\n");
        printf("  ANE:    a'=%.1f, b'=%.1f\n", a_prime, b_prime);

        float diff_a = fabsf(a_prime - 8.0f);
        float diff_b = fabsf(b_prime - (-2.0f));
        CHECK(diff_a < 0.1f && diff_b < 0.1f, "butterfly correct");
    }

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test 2: Butterfly chain (multiple stages)
// This is where the problem emerges - twiddle factors are data-dependent
// ---------------------------------------------------------------------------
static void test_butterfly_chain(int dim) {
    printf("\n=== Test: Butterfly Chain (dim=%d) ===\n", dim);
    printf("  For NTT, each stage has different twiddle factors.\n");
    printf("  ANE can do the add/sub part via conv1x1,\n");
    printf("  but twiddle multiply is element-wise - can't be conv.\n");

    NSString *prog_text = build_butterfly_mil(dim, 1);
    NSData *blob = make_blob_butterfly(dim);
    NSDictionary *wdict = @{@"@model_path/weights/butterfly.bin": @{@"offset": @0, @"data": blob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "bfly_chain");
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    IOSurfaceRef ioX = make_fp32_surface(dim, 1);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, 1);

    // Warmup
    orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);

    // Benchmark
    const int iterations = 1000;
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;
    double ns_per_elem = (ms * 1e6) / dim;

    printf("  Time: %.4f ms/iter, %.2f ns/elem\n", ms, ns_per_elem);

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Analysis: Can NTT twiddle factors be baked into weights?
// ---------------------------------------------------------------------------
static void analyze_twiddle_baking(void) {
    printf("\n=== Analysis: Twiddle Factor Baking ===\n");
    printf("\n");
    printf("  NTT butterfly: (a, b) -> (a + b*w, a - b*w)\n");
    printf("  \n");
    printf("  Can this be expressed as conv?\n");
    printf("  \n");
    printf("  For a single butterfly with fixed w:\n");
    printf("    a' = a + b*w = [1, w] @ [a, b]\n");
    printf("    b' = a - b*w = [1, -w] @ [a, b]\n");
    printf("    \n");
    printf("  Weight matrix: [[1, w], [1, -w]]\n");
    printf("  \n");
    printf("  PROBLEM: Each butterfly position needs DIFFERENT w!\n");
    printf("  \n");
    printf("  In stage s (1-indexed), position j (0-indexed):\n");
    printf("    w = g^(j mod 2^s) for appropriate g\n");
    printf("  \n");
    printf("  Example for N=256, stage 2:\n");
    printf("    butterflies 0,1: w = 1\n");
    printf("    butterflies 2,3: w = g^128\n");
    printf("    butterflies 4,5: w = g^64\n");
    printf("    butterflies 6,7: w = g^192\n");
    printf("    ...\n");
    printf("  \n");
    printf("  ANE conv1x1 applies SAME weight matrix to all positions.\n");
    printf("  Cannot express position-dependent twiddle factors.\n");
    printf("  \n");
    printf("  CONCLUSION: Full NTT cannot be expressed as pure ANE convolutions.\n");
    printf("  \n");
    printf("  POSSIBLE EXCEPTION: If N is small and we unroll the full NTT\n");
    printf("  as a MIL program with hardcoded twiddle weights per stage,\n");
    printf("  but this would require careful MIL scheduling.\n");
}

// ---------------------------------------------------------------------------
// Test 3: CPU NTT benchmark (for comparison)
// ---------------------------------------------------------------------------
static void benchmark_cpu_ntt(void) {
    printf("\n=== CPU NTT Benchmark (BabyBear, N=256) ===\n");

    uint32_t *data = calloc(NTT_SIZE, sizeof(uint32_t));
    for (int i = 0; i < NTT_SIZE; i++) data[i] = i + 1;

    const int iterations = 1000;
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        cpu_ntt_inplace(data, NTT_SIZE, NTT_LOG);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;

    printf("  Time: %.4f ms/iter\n", ms);
    printf("  Throughput: %.0f transforms/sec\n", 1000.0 / ms);

    free(data);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== NTT on ANE Feasibility Study ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        orion_mil_cache_clear();

        // Tests
        test_single_butterfly();
        test_butterfly_chain(4);
        test_butterfly_chain(8);
        test_butterfly_chain(16);
        test_butterfly_chain(32);

        // Analysis
        analyze_twiddle_baking();

        // CPU baseline
        benchmark_cpu_ntt();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        printf("\n");
        printf("=== FINAL CONCLUSION ===\n");
        printf("  NTT on ANE: NOT FEASIBLE for general case.\n");
        printf("  \n");
        printf("  Reason: Twiddle factors are position-dependent,\n");
        printf("  but ANE conv1x1 applies the same weights everywhere.\n");
        printf("  \n");
        printf("  Exception: If the entire NTT can be unrolled as\n");
        printf("  a series of convolutions with pre-baked weights,\n");
        printf("  it might work for very small N (e.g., N <= 16).\n");
        printf("  But for practical sizes (256+), the number of\n");
        printf("  required conv layers would exceed MIL program limits.\n");

        return g_fail > 0 ? 1 : 0;
    }
}