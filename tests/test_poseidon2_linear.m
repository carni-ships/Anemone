// test_poseidon2_linear.m — Poseidon2 Linear Layer on ANE
// Tests Poseidon2's MDS matrix multiplication using ANE Conv1x1.
// S-box (x^5) runs on CPU; linear layer (MDS) runs on ANE.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_poseidon2_linear.m -o test_poseidon2_linear
// Run:
//   ./test_poseidon2_linear

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "mil_cache.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// Poseidon2 MDS Matrix (t=3 state, BN254 Fr)
// Standard MDS for Poseidon2: [[2,1,1],[1,2,1],[1,1,2]] after suitable S-boxes
// For BN254 Fr, we use field representation via fp16 approximation
// ---------------------------------------------------------------------------
// Note: Poseidon2's MDS is applied as a matrix multiply over the field.
// For ANE, we approximate with fp16 weights. This tests whether the LINEAR
// layer can be accelerated, with S-box still on CPU.

// Poseidon2 parameters (t=3, round numbers vary by spec)
// Full rounds: 8, Partial rounds: 56 (typical for BN254)
// MDS matrix (3x3) - entries are field elements
static const float kPoseidon2MDS[3][3] = {
    { 2.0f, 1.0f, 1.0f },
    { 1.0f, 2.0f, 1.0f },
    { 1.0f, 1.0f, 2.0f }
};

// Alternative: Hadamard-style MDS (more ANE-friendly)
// This is the actual MDS used in many Poseidon2 implementations
static const float kPoseidon2MDS_HADAMARD[3][3] = {
    { 0.408248f, 0.408248f, 0.408248f },  // ~1/sqrt(6)
    { 0.577350f, -0.288675f, -0.288675f }, // ~1/sqrt(3), -1/(2*sqrt(3))
    { 0.0f,      0.707107f, -0.707107f },  // ~1/sqrt(2)
};

// ---------------------------------------------------------------------------
// Weight blob for Poseidon2 MDS (3x3 -> expressed as 3x3 conv)
// dim=3 represents state width t=3
// ---------------------------------------------------------------------------
static NSData *make_blob_mds(int t, const float mds[t][t]) {
    int dim = t;
    int ws = dim * dim * 2; // fp16
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            fp16[i * dim + j] = (_Float16)mds[i][j];
        }
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ---------------------------------------------------------------------------
// CPU baseline: Poseidon2 MDS multiplication (naive matmul)
// ---------------------------------------------------------------------------
static void cpu_poseidon2_mds(const float *input, const float mds[3][3], float *output, int t) {
    for (int i = 0; i < t; i++) {
        output[i] = 0;
        for (int j = 0; j < t; j++) {
            output[i] += input[j] * mds[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// S-box: x^5 (CPU only - ANE can't do nonlinear)
// In real Poseidon2, this is x^5 for full rounds, x^2 for partial rounds
// ---------------------------------------------------------------------------
static void cpu_poseidon2_sbox(float *state, int t) {
    for (int i = 0; i < t; i++) {
        // x^5 = ((x^2)^2) * x
        float x2 = state[i] * state[i];
        float x4 = x2 * x2;
        state[i] = x4 * state[i];
    }
}

// ---------------------------------------------------------------------------
// Full Poseidon2 round on CPU (MDS + S-box)
// ---------------------------------------------------------------------------
static void cpu_poseidon2_round(float *state, int t, const float mds[3][3], int is_partial) {
    float after_mds[3];
    cpu_poseidon2_mds(state, mds, after_mds, t);
    for (int i = 0; i < t; i++) state[i] = after_mds[i];
    if (is_partial) {
        // Partial: only first element gets S-box
        float x2 = state[0] * state[0];
        float x4 = x2 * x2;
        state[0] = x4 * state[0];
    } else {
        cpu_poseidon2_sbox(state, t);
    }
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
// Build Poseidon2 MDS MIL program (t=3 state, single MDS application)
// Uses orion_mil_linear which properly sets up conv with BLOBFILE
// ---------------------------------------------------------------------------
static NSString *build_poseidon2_mds_mil(int t, int batch) {
    // For Poseidon2 MDS on ANE:
    // - Input: [1, t, 1, batch] (batch parallel states)
    // - Weight: [t, t, 1, 1] (MDS matrix)
    // - Output: [1, t, 1, batch]

    // Use orion_mil_linear which properly handles the conv setup
    NSString *wpath = @"@model_path/weights/mds.bin";
    NSString *conv_body = orion_mil_linear("mds", "x16", t, t, batch, [wpath UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", t, batch];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = mds_out)[name = string(\"out\")];\n", t, batch];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", t, batch]],
        @"y");
}

// ---------------------------------------------------------------------------
// Test 1: Single MDS application (correctness)
// ---------------------------------------------------------------------------
static void test_mds_correctness(int t, int batch) {
    printf("\n=== MDS Correctness: t=%d, batch=%d ===\n", t, batch);

    NSString *prog_text = build_poseidon2_mds_mil(t, batch);
    NSData *blob = make_blob_mds(t, kPoseidon2MDS);
    NSDictionary *wdict = @{@"@model_path/weights/mds.bin": @{@"offset": @0, @"data": blob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "poseidon2_mds");
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    // Create input surface with sequential values
    IOSurfaceRef ioX = make_fp32_surface(t, batch);
    IOSurfaceRef ioY = orion_tensor_create_f32(t, batch);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        // Compare with CPU baseline
        float cpu_out[3], ane_out[3];
        float input[3];
        for (int i = 0; i < t; i++) input[i] = read_fp32_elem(ioX, i);

        cpu_poseidon2_mds(input, kPoseidon2MDS, cpu_out, t);

        for (int i = 0; i < t; i++) {
            ane_out[i] = read_fp32_elem(ioY, i);
        }

        printf("  Input:  [%.3f, %.3f, %.3f]\n", input[0], input[1], input[2]);
        printf("  CPU:    [%.3f, %.3f, %.3f]\n", cpu_out[0], cpu_out[1], cpu_out[2]);
        printf("  ANE:    [%.3f, %.3f, %.3f]\n", ane_out[0], ane_out[1], ane_out[2]);

        float max_diff = 0;
        for (int i = 0; i < t; i++) {
            float diff = fabsf(ane_out[i] - cpu_out[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  Max diff: %.6f\n", max_diff);
        // Allow larger tolerance for fp16 approximation
        CHECK(max_diff < 0.1f, "ANE matches CPU (within fp16 tolerance)");
    }

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test 2: Batch MDS (throughput)
// ---------------------------------------------------------------------------
static void test_mds_batch_throughput(int t, int batch, int iterations) {
    printf("\n=== MDS Batch Throughput: t=%d, batch=%d, iter=%d ===\n", t, batch, iterations);

    NSString *prog_text = build_poseidon2_mds_mil(t, batch);
    NSData *blob = make_blob_mds(t, kPoseidon2MDS);
    NSDictionary *wdict = @{@"@model_path/weights/mds.bin": @{@"offset": @0, @"data": blob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "poseidon2_batch");
    if (!prog) {
        printf("  FAIL: compile failed\n");
        g_fail++;
        return;
    }

    IOSurfaceRef ioX = make_fp32_surface(t, batch);
    IOSurfaceRef ioY = orion_tensor_create_f32(t, batch);

    // Warmup
    orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;
    double ns_per_elem = (ms * 1e6) / (t * batch);

    printf("  Time: %.4f ms/iter\n", ms);
    printf("  Elements: %d, ns/elem: %.2f\n", t * batch, ns_per_elem);

    // Compare with CPU
    float input[3] = {1.0f, 2.0f, 3.0f};
    float cpu_out[3];

    clock_t cpu_start = clock();
    for (int i = 0; i < iterations * 1000; i++) {
        cpu_poseidon2_mds(input, kPoseidon2MDS, cpu_out, t);
    }
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / (iterations * 1000);

    printf("  CPU time: %.6f ms/iter\n", cpu_ms);
    printf("  ANE speedup: %.1fx (overhead-dominated for tiny ops)\n", cpu_ms / ms);

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test 3: Simulated Poseidon2 round (ANE MDS + CPU S-box)
// ---------------------------------------------------------------------------
static void test_poseidon2_round(int t, int batch) {
    printf("\n=== Poseidon2 Round (ANE MDS + CPU S-box): t=%d, batch=%d ===\n", t, batch);

    NSString *prog_text = build_poseidon2_mds_mil(t, batch);
    NSData *blob = make_blob_mds(t, kPoseidon2MDS);
    NSDictionary *wdict = @{@"@model_path/weights/mds.bin": @{@"offset": @0, @"data": blob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "poseidon2_round");
    if (!prog) {
        printf("  FAIL: compile failed\n");
        g_fail++;
        return;
    }

    // Initialize state (simulated)
    float state[3] = {1.0f, 2.0f, 3.0f};
    printf("  Initial state: [%.3f, %.3f, %.3f]\n", state[0], state[1], state[2]);

    // Write state to IOSurface
    IOSurfaceRef ioX = orion_tensor_create_f32(t, batch);
    IOSurfaceRef ioY = orion_tensor_create_f32(t, batch);

    for (int i = 0; i < t; i++) {
        write_fp32_elem(ioX, i, state[i]);
    }

    // ANE: MDS
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "MDS eval succeeds");

    if (ok) {
        for (int i = 0; i < t; i++) {
            state[i] = read_fp32_elem(ioY, i);
        }
        printf("  After MDS: [%.3f, %.3f, %.3f]\n", state[0], state[1], state[2]);

        // CPU: S-box (x^5)
        cpu_poseidon2_sbox(state, t);
        printf("  After S-box: [%.3f, %.3f, %.3f]\n", state[0], state[1], state[2]);

        // Verify round is consistent
        CHECK(isfinite(state[0]) && isfinite(state[1]) && isfinite(state[2]), "state is finite");
    }

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Poseidon2 Linear Layer (MDS) on ANE ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        orion_mil_cache_clear();

        // Test 1: Correctness
        test_mds_correctness(3, 1);    // Single state
        test_mds_correctness(3, 8);    // Batch of 8
        test_mds_correctness(3, 64);   // Batch of 64

        // Test 2: Throughput
        test_mds_batch_throughput(3, 1, 1000);
        test_mds_batch_throughput(3, 8, 500);
        test_mds_batch_throughput(3, 64, 200);
        test_mds_batch_throughput(3, 256, 100);

        // Test 3: Full round simulation
        test_poseidon2_round(3, 1);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}