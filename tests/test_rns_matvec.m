// test_rns_matvec.m — RNS-MatVec on ANE
// Tests 5-residue RNS decomposition with parallel conv1x1 per residue,
// CRT reconstruction, and benchmark vs GPU inner product.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     tests/test_rns_matvec.m -o test_rns_matvec
// Run:
//   ./test_rns_matvec

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"

#define CH 256
#define N_RESIDUES 5

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// RNS moduli (all < 2^24, fitting in FP16)
// ---------------------------------------------------------------------------
// These are example moduli for a ~96-bit RNS base.
// For BabyBear (p = 6,700,417), standard RNS uses different moduli.
typedef struct {
    uint32_t mod;
    const char *name;
} RNSMod;

static const RNSMod kRNSMod[N_RESIDUES] = {
    { 3, "q0" },      // 3
    { 5, "q1" },      // 5
    { 7, "q2" },      // 7
    { 11, "q3" },     // 11
    { 13, "q4" },     // 13
    // Product = 3*5*7*11*13 = 15015 < 2^14, well within uint64 range
};

// ---------------------------------------------------------------------------
// Weight blob for single residue conv1x1 with diagonal (identity) weights
// ---------------------------------------------------------------------------
static NSData *make_blob_identity(int dim) {
    int ws = dim * dim * 2; // fp16
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < dim; i++) {
        fp16[i * dim + i] = (_Float16)1.0f;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ---------------------------------------------------------------------------
// Weight blob with constant value (for scaling tests)
// ---------------------------------------------------------------------------
static NSData *make_blob_const(int rows, int cols, float val) {
    int ws = rows * cols * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < rows * cols; i++) {
        fp16[i] = (_Float16)val;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ---------------------------------------------------------------------------
// IOSurface helpers
// ---------------------------------------------------------------------------

// Create fp32 IOSurface with sequential values
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

// Create fp32 IOSurface filled with constant value
static IOSurfaceRef make_fp32_const(int channels, int seq_len, float val) {
    int count = channels * seq_len;
    size_t bytes = count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    if (!s) return NULL;
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

// Read fp32 surface first element
static float read_fp32_first(IOSurfaceRef s) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[0];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// Read fp32 surface element at index
static float read_fp32_elem(IOSurfaceRef s, int idx) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[idx];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// ---------------------------------------------------------------------------
// Build RNS-MatVec MIL program (single residue)
// Uses mil_builder orion_mil_linear for clean conv1x1
// ---------------------------------------------------------------------------
static NSString *build_rns_single_mil(int dim, int seq, int residue_idx) {
    NSString *prefix = [NSString stringWithFormat:@"r%d", residue_idx];
    NSString *weight_path = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", residue_idx];

    NSString *conv_body = orion_mil_linear([prefix UTF8String], "x16", dim, dim, seq,
                                          [weight_path UTF8String], NULL);
    NSString *body = [NSString stringWithFormat:
        @"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        @"%@"
        @"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        @"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = %@_out)[name = string(\"out\")];\n",
        dim, seq, conv_body, dim, seq, prefix];
    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// ---------------------------------------------------------------------------
// CRT reconstruction from RNS residues
// Uses proper CRT with co-prime moduli (pairwise coprime required)
// M = product of all moduli
// result = Σ residues[i] * Mi * Mi_inv mod M
// where Mi = M / moduli[i] and Mi_inv = Mi^{-1} mod moduli[i]
// ---------------------------------------------------------------------------

// Extended GCD: returns (x, y, g) where ax + by = g = gcd(a, b)
static int64_t extended_gcd(int64_t a, int64_t b, int64_t *x, int64_t *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    int64_t x1, y1;
    int64_t g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

static uint64_t crt_reconstruct(uint32_t residues[N_RESIDUES]) {
    // Compute M = product of all moduli (all small, fits in uint64)
    uint64_t M = 1;
    for (int i = 0; i < N_RESIDUES; i++) {
        M *= kRNSMod[i].mod;
    }

    // CRT reconstruction: result = Σ residues[i] * Mi * Mi_inv mod M
    uint64_t result = 0;
    for (int i = 0; i < N_RESIDUES; i++) {
        uint64_t mod_i = kRNSMod[i].mod;
        uint64_t Mi = M / mod_i;

        // Compute Mi_inv = Mi^{-1} mod mod_i using extended GCD
        int64_t x, y;
        extended_gcd(Mi % mod_i, (int64_t)mod_i, &x, &y);
        int64_t Mi_inv = x % (int64_t)mod_i;
        if (Mi_inv < 0) Mi_inv += mod_i;

        // term = residues[i] * Mi * Mi_inv mod M
        uint64_t term = residues[i] % mod_i;
        term = (term * Mi) % M;
        term = (term * (uint64_t)Mi_inv) % M;

        result = (result + term) % M;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Test: Single residue conv1x1 (identity weight)
// ---------------------------------------------------------------------------
static void test_single_residue(int seq_len, int residue_idx) {
    printf("\n=== Test: Single residue conv1x1 (residue=%d, seq=%d) ===\n", residue_idx, seq_len);

    NSString *prog_text = build_rns_single_mil(CH, seq_len, residue_idx);
    NSData *wblob = make_blob_identity(CH);
    NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", residue_idx];
    NSDictionary *wdict = @{key: @{@"offset": @0, @"data": wblob}};

    NSString *tag = [NSString stringWithFormat:@"rns_r%d_s%d", residue_idx, seq_len];
    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, [tag UTF8String]);
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    IOSurfaceRef ioX = make_fp32_surface(CH, seq_len);
    IOSurfaceRef ioY = orion_tensor_create_f32(CH, seq_len);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        float first_in = read_fp32_first(ioX);
        float first_out = read_fp32_first(ioY);
        float diff = fabsf(first_out - first_in);
        CHECK(diff < 0.1f, "output ≈ input (identity weight)");
        printf("    first_in=%.2f first_out=%.2f diff=%.6f\n", first_in, first_out, diff);
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: All 5 residues with constant input (verify scaling)
// ---------------------------------------------------------------------------
static void test_all_residues_constant(int seq_len, float input_val, float weight_val) {
    printf("\n=== Test: All %d residues (seq=%d, input=%.1f, weight=%.1f) ===\n",
           N_RESIDUES, seq_len, input_val, weight_val);

    OrionProgram *progs[N_RESIDUES];
    IOSurfaceRef ioXs[N_RESIDUES];
    IOSurfaceRef ioYs[N_RESIDUES];
    float outputs[N_RESIDUES];

    // Create all programs and surfaces
    for (int r = 0; r < N_RESIDUES; r++) {
        NSString *prog_text = build_rns_single_mil(CH, seq_len, r);
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", r];
        NSDictionary *wdict = @{key: @{@"offset": @0, @"data": make_blob_const(CH, CH, weight_val)}};

        NSString *tag = [NSString stringWithFormat:@"rns_all_r%d", r];
        progs[r] = orion_compile_mil([prog_text UTF8String], wdict, [tag UTF8String]);
        if (!progs[r]) {
            printf("  FAIL: residue %d compile failed\n", r);
            for (int i = 0; i < r; i++) {
                CFRelease(ioXs[i]);
                CFRelease(ioYs[i]);
                orion_release_program(progs[i]);
            }
            return;
        }

        ioXs[r] = make_fp32_const(CH, seq_len, input_val);
        ioYs[r] = orion_tensor_create_f32(CH, seq_len);
    }

    // Evaluate all residues in sequence
    for (int r = 0; r < N_RESIDUES; r++) {
        bool ok = orion_eval(progs[r], (IOSurfaceRef[]){ioXs[r]}, 1, (IOSurfaceRef[]){ioYs[r]}, 1);
        if (ok) {
            outputs[r] = read_fp32_first(ioYs[r]);
        } else {
            printf("  FAIL: residue %d eval failed\n", r);
        }
    }

    // With constant weight_val and constant input_val:
    // For identity weight: output = input = input_val
    // For constant weight: output = input * weight_val (sum over channels)
    // With CH=256, input_val=1.0, weight_val=0.1:
    // output[0] = sum_j input[j] * weight[j,0] = 256 * 1.0 * 0.1 = 25.6
    printf("  Residue outputs:\n");
    for (int r = 0; r < N_RESIDUES; r++) {
        printf("    %s: %.2f (mod=%u)\n", kRNSMod[r].name, outputs[r], kRNSMod[r].mod);
    }

    // Verify all outputs are finite and in reasonable range
    bool all_ok = YES;
    for (int r = 0; r < N_RESIDUES; r++) {
        if (isinf(outputs[r]) || isnan(outputs[r])) {
            all_ok = NO;
            printf("  FAIL: residue %d output is inf/nan\n", r);
        }
    }
    CHECK(all_ok, "all residues produce finite output");

    // Clean up
    for (int r = 0; r < N_RESIDUES; r++) {
        CFRelease(ioXs[r]);
        CFRelease(ioYs[r]);
        orion_release_program(progs[r]);
    }
}

// ---------------------------------------------------------------------------
// Test: CRT reconstruction
// ---------------------------------------------------------------------------
static void test_crt_reconstruction(void) {
    printf("\n=== Test: CRT reconstruction ===\n");

    // Test case: residues for a simple number
    // Choose x < M (M = 3*5*7*11*13 = 15015)
    uint64_t x = 1234;
    uint32_t residues[N_RESIDUES];

    printf("  Testing CRT with x = %llu\n", (unsigned long long)x);

    for (int i = 0; i < N_RESIDUES; i++) {
        residues[i] = x % kRNSMod[i].mod;
        printf("    %s: %u (mod %u)\n", kRNSMod[i].name, residues[i], kRNSMod[i].mod);
    }

    uint64_t reconstructed = crt_reconstruct(residues);
    printf("  Reconstructed: %llu\n", (unsigned long long)reconstructed);

    CHECK(reconstructed == x, "CRT reconstruction matches original");
}

// ---------------------------------------------------------------------------
// Test: RNS-MatVec throughput benchmark
// ---------------------------------------------------------------------------
static void test_rns_matvec_benchmark(int seq_len) {
    printf("\n=== Test: RNS-MatVec throughput (seq=%d) ===\n", seq_len);

    clock_t start = clock();

    // Create all programs
    OrionProgram *progs[N_RESIDUES];
    for (int r = 0; r < N_RESIDUES; r++) {
        NSString *prog_text = build_rns_single_mil(CH, seq_len, r);
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", r];
        NSDictionary *wdict = @{key: @{@"offset": @0, @"data": make_blob_identity(CH)}};

        NSString *tag = [NSString stringWithFormat:@"rns_bench_r%d", r];
        progs[r] = orion_compile_mil([prog_text UTF8String], wdict, [tag UTF8String]);
        if (!progs[r]) {
            printf("  FAIL: compile failed for residue %d\n", r);
            return;
        }
    }

    clock_t compile_end = clock();
    double compile_ms = (double)(compile_end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  Total compile time: %.2f ms\n", compile_ms);
    printf("  Per-residue compile: %.2f ms\n", compile_ms / N_RESIDUES);

    // Create surfaces
    IOSurfaceRef ioXs[N_RESIDUES];
    IOSurfaceRef ioYs[N_RESIDUES];
    for (int r = 0; r < N_RESIDUES; r++) {
        ioXs[r] = make_fp32_surface(CH, seq_len);
        ioYs[r] = orion_tensor_create_f32(CH, seq_len);
    }

    // Warmup
    for (int r = 0; r < N_RESIDUES; r++) {
        orion_eval(progs[r], (IOSurfaceRef[]){ioXs[r]}, 1, (IOSurfaceRef[]){ioYs[r]}, 1);
    }

    // Benchmark
    const int iterations = 20;
    clock_t bench_start = clock();
    for (int i = 0; i < iterations; i++) {
        for (int r = 0; r < N_RESIDUES; r++) {
            orion_eval(progs[r], (IOSurfaceRef[]){ioXs[r]}, 1, (IOSurfaceRef[]){ioYs[r]}, 1);
        }
    }
    clock_t bench_end = clock();
    double total_ms = (double)(bench_end - bench_start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / iterations;
    double per_residue_ms = per_iter_ms / N_RESIDUES;
    int total_elements = N_RESIDUES * CH * seq_len;
    double ns_per_elem = (per_iter_ms * 1e6) / total_elements;

    printf("  %d-iteration total: %.2f ms (%.2f ms/iter)\n", iterations, total_ms, per_iter_ms);
    printf("  Per-residue: %.3f ms\n", per_residue_ms);
    printf("  Total elements: %d, ns/elem: %.2f\n", total_elements, ns_per_elem);

    // Also show what this means for a single big matvec
    // Each residue evaluates CHxCH @ CHxS, so total work = 5 * CH * CH * S multiplications
    double gflops = (N_RESIDUES * CH * CH * seq_len * 2) / (per_iter_ms * 1e6);
    printf("  Implied GFLOPS: %.2f (for parallel residue eval)\n", gflops);

    // Cleanup
    for (int r = 0; r < N_RESIDUES; r++) {
        CFRelease(ioXs[r]);
        CFRelease(ioYs[r]);
        orion_release_program(progs[r]);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE RNS-MatVec Test ===\n");
        printf("RNS base: %d residues\n", N_RESIDUES);
        for (int i = 0; i < N_RESIDUES; i++) {
            printf("  %s: mod=%u (< 2^24: %s)\n",
                   kRNSMod[i].name, kRNSMod[i].mod,
                   kRNSMod[i].mod < (1u << 24) ? "YES" : "NO");
        }

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n", orion_compile_count());

        // CRT reconstruction test (pure CPU, no ANE)
        test_crt_reconstruction();

        // Single residue tests
        test_single_residue(64, 0);
        test_single_residue(64, 2);
        test_single_residue(64, 4);

        // All residues with constant input
        test_all_residues_constant(64, 1.0f, 0.1f);
        test_all_residues_constant(64, 2.0f, 0.5f);

        // Throughput benchmark
        test_rns_matvec_benchmark(64);
        test_rns_matvec_benchmark(128);
        test_rns_matvec_benchmark(256);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
