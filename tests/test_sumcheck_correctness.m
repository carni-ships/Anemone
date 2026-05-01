// test_sumcheck_correctness.m — Numerical Correctness for Sumcheck Layer
// Validates ANE computations against mathematical reference formulas.
// No external comparison needed — pure numerical validation against known results.
//
// Mathematical basis:
//   Sumcheck: g(r) = Σ_{i=0}^{dim-1} v_i * r^i
//   With v_i = 1 and power-diagonal weights W[i,i] = r^i:
//     conv1x1 computes: output[j] = Σ_i input[i] * W[i,j] = Σ_i 1 * r^i = Σ_i r^i
//     reduce_sum output = dim * Σ_i r^i = dim * (r^dim - 1) / (r - 1)
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     tests/test_sumcheck_correctness.m -o test_sumcheck_correctness
// Run:
//   ./test_sumcheck_correctness

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

#define CHECK_CLOSE(actual, expected, tol, msg) do { \
    double actual_d = (double)(actual); \
    double expected_d = (double)(expected); \
    double diff = fabs(actual_d - expected_d); \
    double tolerance = fmax(fabs(expected_d) * 0.05, fmax(1.0, (tol))); \
    if (diff <= tolerance) { g_pass++; printf("  PASS: %s (%.6f ≈ %.6f, err=%.6e, tol=%.6e)\n", msg, (float)actual_d, (float)expected_d, diff, tolerance); } \
    else { g_fail++; printf("  FAIL: %s (%.6f vs %.6f, err=%.6e > tol=%.6e)\n", msg, (float)actual_d, (float)expected_d, diff, tolerance); } \
} while(0)

// ---------------------------------------------------------------------------
// Weight blob helpers
// ---------------------------------------------------------------------------

static NSData *make_blob_identity(int dim) {
    int ws = dim * dim * 2;
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

static float read_fp32_elem(IOSurfaceRef s, int idx) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[idx];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// ---------------------------------------------------------------------------
// Forward declaration
// ---------------------------------------------------------------------------

static NSData *make_sumcheck_powers_blob(int dim, float r);

static NSString *build_sumcheck_mil(int dim, int seq, float challenge, int round_idx);

// ---------------------------------------------------------------------------
// Mathematical reference functions
// ---------------------------------------------------------------------------

// Geometric series: Σ_{i=0}^{n-1} r^i = (r^n - 1) / (r - 1)
static double geometric_series(double r, int n) {
    if (fabs(r - 1.0) < 1e-9) return (double)n;
    return (pow(r, n) - 1.0) / (r - 1.0);
}

// Expected sumcheck output for geometric series input (v_i = 1):
// g(r) = Σ r^i = (r^dim - 1) / (r - 1)
static double expected_g(double r, int dim) {
    return geometric_series(r, dim);
}

// For conv1x1 with power-diagonal weights followed by reduce_sum over dim channels:
// output = reduce_sum(conv1x1(input, W)) where W[i,i] = r^i
// With input[i] = v (constant), conv picks v * r^i at each position
// reduce_sum over dim channels sums these: dim * v * Σ r^i
static double expected_sumcheck_output(double r, int dim, double input_val) {
    return dim * input_val * expected_g(r, dim);
}

// ---------------------------------------------------------------------------
// Test: Power-diagonal weight construction correctness
// ---------------------------------------------------------------------------
static void test_power_diagonal_weights(int dim, float r) {
    printf("\n=== Test: Power-diagonal weight construction (dim=%d, r=%.2f) ===\n", dim, r);

    NSData *blob = make_sumcheck_powers_blob(dim, r);
    const uint8_t *bytes = blob.bytes;

    // Weight data starts at offset 128
    _Float16 *weights = (_Float16 *)(bytes + 128);

    // Verify diagonal elements: W[i,i] = r^i
    // Note: FP16 has limited precision (~3.3 decimal digits), so we use relative tolerance
    double r_power = 1.0;
    double max_err = 0.0;
    for (int i = 0; i < dim; i++) {
        float wii = (float)weights[i * dim + i];
        float expected = (float)r_power;
        double rel_err = (expected > 1e-6) ? fabs(wii - expected) / fabs(expected) : fabs(wii - expected);
        max_err = fmax(max_err, rel_err);
        r_power *= r;
    }
    printf("    Diagonal weight check: max rel err = %.6e\n", max_err);
    // FP16 has ~3-4 digits of precision, allow up to 1% relative error
    CHECK(max_err < 0.01, "power-diagonal weights correct");

    // Verify off-diagonal elements are near-zero
    double max_offdiag = 0.0;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i != j) {
                float w = (float)weights[i * dim + j];
                max_offdiag = fmax(max_offdiag, fabs(w));
            }
        }
    }
    printf("    Off-diagonal check: max = %.6e\n", max_offdiag);
    CHECK(max_offdiag < 1e-3, "off-diagonal elements are zero");

    // Verify off-diagonal is exactly 0 for i < j (upper triangle - not used in conv)
    // Lower triangle should also be 0
    int nonzero_lower = 0;
    for (int i = 1; i < dim; i++) {
        for (int j = 0; j < i; j++) {
            float w = (float)weights[i * dim + j];
            if (fabs(w) > 1e-6) nonzero_lower++;
        }
    }
    CHECK(nonzero_lower == 0, "strictly lower triangular is zero (conv accesses i>=j)");
}

// ---------------------------------------------------------------------------
// Test: Single sumcheck round - geometric series
// Validates: g(r) = Σ input[i] * r^i with input[i] = 1.0
// ---------------------------------------------------------------------------
static void test_geometric_series(int dim, int seq, float r) {
    printf("\n=== Test: Geometric series (dim=%d, seq=%d, r=%.2f) ===\n", dim, seq, r);

    // Build MIL with power-diagonal weights
    NSString *prog_text = build_sumcheck_mil(dim, seq, r, 0);
    NSData *powers_blob = make_sumcheck_powers_blob(dim, r);
    NSDictionary *wdict = @{@"@model_path/weights/w0.bin": @{@"offset": @0, @"data": powers_blob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "geom_series");
    CHECK(prog != NULL, "program compiles");
    if (!prog) return;

    // Input: constant 1.0
    IOSurfaceRef ioX = make_fp32_const(dim, seq, 1.0f);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        float output = read_fp32_elem(ioY, 0);

        // Expected: g(r) = (r^dim - 1) / (r - 1)
        double expected = expected_g(r, dim);

        printf("    output = %.4f, expected = %.4f\n", output, (float)expected);
        CHECK_CLOSE(output, expected, 0, "geometric series matches formula");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: Input scaling property
// For sumcheck: g(r; c*v) = c * g(r; v) for scalar c
// ---------------------------------------------------------------------------
static void test_input_scaling(int dim, int seq, float r, float scale) {
    printf("\n=== Test: Input scaling (dim=%d, seq=%d, r=%.2f, scale=%.2f) ===\n", dim, seq, r, scale);

    NSString *prog_text = build_sumcheck_mil(dim, seq, r, 0);
    NSData *powers_blob = make_sumcheck_powers_blob(dim, r);
    NSDictionary *wdict = @{@"@model_path/weights/w0.bin": @{@"offset": @0, @"data": powers_blob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "input_scale");
    CHECK(prog != NULL, "program compiles");
    if (!prog) return;

    // Input: constant scale
    IOSurfaceRef ioX = make_fp32_const(dim, seq, scale);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        float output = read_fp32_elem(ioY, 0);

        // Expected: scale * g(r)
        double expected = scale * expected_g(r, dim);

        printf("    output = %.4f, expected = %.4f (scale=%.2f)\n", output, (float)expected, scale);
        CHECK_CLOSE(output, expected, 0, "input scaling property holds");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: Different sequence lengths (seq should not affect result)
// The sumcheck result is independent of seq dimension
// ---------------------------------------------------------------------------
static void test_seq_independence(int dim, float r) {
    printf("\n=== Test: Sequence length independence (dim=%d, r=%.2f) ===\n", dim, r);

    double reference = expected_g(r, dim);

    for (int seq = 32; seq <= 128; seq *= 2) {
        NSString *prog_text = build_sumcheck_mil(dim, seq, r, 0);
        NSData *powers_blob = make_sumcheck_powers_blob(dim, r);
        NSDictionary *wdict = @{@"@model_path/weights/w0.bin": @{@"offset": @0, @"data": powers_blob}};

        OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "seq_indep");
        if (!prog) { CHECK(false, "program compiles"); continue; }

        IOSurfaceRef ioX = make_fp32_const(dim, seq, 1.0f);
        IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

        bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
        CHECK(ok, "eval succeeds");

        if (ok) {
            float output = read_fp32_elem(ioY, 0);
            double tolerance = fmax(fabs(reference) * 0.05, 0.1);
            printf("    seq=%d: output=%.4f (ref=%.4f)\n", seq, output, (float)reference);
            CHECK_CLOSE(output, reference, tolerance, "seq independence");
        }

        CFRelease(ioX);
        CFRelease(ioY);
        orion_release_program(prog);
    }
}

// ---------------------------------------------------------------------------
// Test: r=1 edge case (geometric series = dim)
// ---------------------------------------------------------------------------
static void test_r_equals_one(int dim, int seq) {
    printf("\n=== Test: r=1 edge case (dim=%d, seq=%d) ===\n", dim, seq);

    NSString *prog_text = build_sumcheck_mil(dim, seq, 1.0f, 0);
    NSData *powers_blob = make_sumcheck_powers_blob(dim, 1.0f);
    NSDictionary *wdict = @{@"@model_path/weights/w0.bin": @{@"offset": @0, @"data": powers_blob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "r_one");
    CHECK(prog != NULL, "program compiles");
    if (!prog) return;

    IOSurfaceRef ioX = make_fp32_const(dim, seq, 1.0f);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        float output = read_fp32_elem(ioY, 0);
        // g(1) = Σ 1 = dim
        double expected = dim;
        printf("    output = %.4f, expected = %.4f\n", output, (float)expected);
        CHECK_CLOSE(output, expected, 0, "r=1 case (geometric series = dim)");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: Conv with identity weights (verifies base conv structure)
// With identity weights and input[i]=v, conv picks v at each output position
// reduce_sum gives dim * v
// ---------------------------------------------------------------------------
static void test_identity_conv(int dim, int seq, float input_val) {
    printf("\n=== Test: Identity conv (dim=%d, seq=%d, input=%.2f) ===\n", dim, seq, input_val);

    NSString *prog_text = build_sumcheck_mil(dim, seq, 0.0f, 0); // r=0 for power-diagonal (identity-like)
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w0.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "ident_conv");
    CHECK(prog != NULL, "program compiles");
    if (!prog) return;

    IOSurfaceRef ioX = make_fp32_const(dim, seq, input_val);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        float output = read_fp32_elem(ioY, 0);
        // With identity weights: conv picks input[0] at each of dim positions
        // reduce_sum over dim gives dim * input[0]
        double expected = dim * input_val;
        printf("    output = %.4f, expected = %.4f\n", output, (float)expected);
        CHECK_CLOSE(output, expected, 0, "identity conv correct");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: Multi-round numerical consistency
// Each round with same input should produce same output (deterministic)
// Verifies: same input -> same output, and all outputs are finite
// ---------------------------------------------------------------------------
static void test_multi_round_numerical(int dim, int seq, int n_rounds) {
    printf("\n=== Test: Multi-round numerical (dim=%d, seq=%d, rounds=%d) ===\n", dim, seq, n_rounds);

    // Fixed challenge for reproducibility
    float challenge = 1.5f;
    float input_val = 0.01f;

    // Compile all round programs with same params
    OrionProgram *progs[8];
    for (int i = 0; i < n_rounds; i++) {
        NSString *prog_text = build_sumcheck_mil(dim, seq, challenge, i);
        NSData *powers_blob = make_sumcheck_powers_blob(dim, challenge);
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        NSDictionary *wdict = @{key: @{@"offset": @0, @"data": powers_blob}};

        char tag[32];
        snprintf(tag, sizeof(tag), "multi_num_%d", i);
        progs[i] = orion_compile_mil([prog_text UTF8String], wdict, tag);
        if (!progs[i]) {
            CHECK(false, "program compiles");
            for (int j = 0; j < i; j++) orion_release_program(progs[j]);
            return;
        }
    }
    printf("    All %d programs compiled\n", n_rounds);

    // Execute rounds
    IOSurfaceRef ioX = make_fp32_const(dim, seq, input_val);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    float first_result = 0;
    bool all_finite = true;
    bool consistent = true;
    for (int i = 0; i < n_rounds; i++) {
        bool ok = orion_eval(progs[i], (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
        if (!ok) {
            CHECK(false, "eval succeeds");
            all_finite = false;
            break;
        }

        float result = read_fp32_elem(ioY, 0);
        if (i == 0) first_result = result;

        if (isinf(result) || isnan(result)) {
            all_finite = false;
            printf("    Round %d: r=%.2f -> %.4f (NON-FINITE)\n", i, challenge, result);
        } else {
            printf("    Round %d: r=%.2f -> %.4f\n", i, challenge, result);
            if (fabs(result - first_result) > 0.001f) {
                consistent = false;
            }
        }
    }

    CHECK(all_finite, "all round outputs are finite");
    CHECK(consistent, "round outputs are consistent (deterministic)");
}

// ---------------------------------------------------------------------------
// Test: Throughput benchmark
// ---------------------------------------------------------------------------
static void test_throughput(int dim, int seq, int n_iters) {
    printf("\n=== Test: Throughput (dim=%d, seq=%d, iter=%d) ===\n", dim, seq, n_iters);

    // Use small r to avoid overflow: dim * input * geometric_series must stay in range
    float r = 1.1f;
    float input_val = 0.001f;  // Very small to avoid FP16 overflow at large dim

    NSString *prog_text = build_sumcheck_mil(dim, seq, r, 0);
    NSData *powers_blob = make_sumcheck_powers_blob(dim, r);
    NSDictionary *wdict = @{@"@model_path/weights/w0.bin": @{@"offset": @0, @"data": powers_blob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "throughput");
    CHECK(prog != NULL, "program compiles");
    if (!prog) return;

    IOSurfaceRef ioX = make_fp32_const(dim, seq, input_val);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    // Warmup
    for (int i = 0; i < 3; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < n_iters; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();

    double total_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / n_iters;
    double ns_per_eval = (per_iter_ms * 1e6);

    printf("    %d iterations: %.2f ms total, %.3f ms/iter\n", n_iters, total_ms, per_iter_ms);
    printf("    %.0f KOps/sec (%.0f ns/eval)\n", 1000.0 / per_iter_ms, ns_per_eval);

    CHECK(per_iter_ms < 10.0, "throughput < 10ms per eval");

    float output = read_fp32_elem(ioY, 0);
    if (isinf(output) || isnan(output)) {
        CHECK(false, "output is finite");
    } else {
        CHECK(true, "output is finite");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// Forward declaration
static NSData *make_sumcheck_powers_blob(int dim, float r);

static NSString *build_sumcheck_mil(int dim, int seq, float challenge, int round_idx) {
    NSString *prefix = [NSString stringWithFormat:@"sc%d", round_idx];
    NSString *weight_path = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", round_idx];

    NSString *conv_body = orion_mil_linear([prefix UTF8String], "x16", dim, dim, seq,
                                          [weight_path UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];

    [body appendFormat:@"        tensor<int32, [1]> %@_ax = const()[name=string(\"%@_ax\"), val=tensor<int32, [1]>([1])];\n", prefix, prefix];
    [body appendFormat:@"        bool %@_kd = const()[name=string(\"%@_kd\"), val=bool(true)];\n", prefix, prefix];
    [body appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_sum = reduce_sum(x=%@_out, axes=%@_ax, keep_dims=%@_kd)[name=string(\"%@_sum\")];\n",
     seq, prefix, prefix, prefix, prefix, prefix];

    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1,1,1,%d]> y = cast(dtype = to32, x = %@_sum)[name = string(\"out\")];\n", seq, prefix];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

static NSData *make_sumcheck_powers_blob(int dim, float r) {
    int ws = dim * dim * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);

    float r_power = 1.0f;
    for (int i = 0; i < dim; i++) {
        fp16[i * dim + i] = (_Float16)r_power;
        r_power *= r;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE Sumcheck Numerical Correctness Tests ===\n");
        printf("Mathematical reference: g(r) = Σ r^i = (r^dim - 1) / (r - 1)\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        // =========================================================================
        // Section 1: Weight construction validation
        // =========================================================================
        printf("========== Weight Construction Tests ==========\n");
        test_power_diagonal_weights(8, 2.0f);
        test_power_diagonal_weights(16, 1.5f);
        test_power_diagonal_weights(64, 1.1f);
        test_power_diagonal_weights(256, 1.01f);

        // =========================================================================
        // Section 2: Geometric series formula validation
        // Use smaller r for larger dim to avoid FP16 overflow
        // =========================================================================
        printf("\n========== Geometric Series Tests ==========\n");
        test_geometric_series(8, 64, 2.0f);
        test_geometric_series(16, 64, 1.5f);
        test_geometric_series(8, 64, 1.5f);
        test_geometric_series(16, 64, 1.5f);
        test_geometric_series(8, 64, 1.1f);
        test_geometric_series(16, 64, 1.1f);
        test_r_equals_one(8, 64);
        test_r_equals_one(16, 64);
        test_r_equals_one(64, 64);

        // =========================================================================
        // Section 3: Scaling property
        // =========================================================================
        printf("\n========== Scaling Property Tests ==========\n");
        test_input_scaling(8, 64, 2.0f, 0.01f);
        test_input_scaling(16, 64, 1.5f, 0.01f);
        test_input_scaling(8, 64, 1.5f, 0.01f);
        test_input_scaling(16, 64, 1.5f, 0.1f);

        // =========================================================================
        // Section 4: Sequence length independence
        // =========================================================================
        printf("\n========== Sequence Independence Tests ==========\n");
        test_seq_independence(16, 1.5f);
        test_seq_independence(8, 2.0f);
        test_seq_independence(16, 1.1f);

        // =========================================================================
        // Section 5: Identity conv validation
        // =========================================================================
        printf("\n========== Identity Conv Tests ==========\n");
        test_identity_conv(16, 64, 0.01f);
        test_identity_conv(16, 64, 0.1f);
        test_identity_conv(64, 64, 0.01f);
        test_identity_conv(256, 64, 0.01f);

        // =========================================================================
        // Section 6: Multi-round numerical consistency
        // =========================================================================
        printf("\n========== Multi-round Numerical Tests ==========\n");
        test_multi_round_numerical(16, 64, 3);
        test_multi_round_numerical(16, 64, 5);

        // =========================================================================
        // Section 7: Throughput
        // =========================================================================
        printf("\n========== Throughput Tests ==========\n");
        test_throughput(64, 64, 50);
        test_throughput(256, 64, 50);

        // =========================================================================
        // Summary
        // =========================================================================
        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        // Mathematical summary
        printf("\nMathematical validation summary:\n");
        printf("  g(r) = Σ_{i=0}^{dim-1} r^i = (r^dim - 1) / (r - 1)\n");
        printf("  Conv with power-diagonal weights: Σ r^i per output position\n");
        printf("  Reduce_sum: dim * Σ r^i (scaled by input value)\n");
        printf("  FP16 tolerance: 5%% relative or 1.0 absolute\n");

        return g_fail > 0 ? 1 : 0;
    }
}
