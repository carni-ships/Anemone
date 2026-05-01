// test_tensor_proof_matvec.m — Tensor Proof MatVec on ANE
// Implements matrix-vector multiply for Spartan/Spark tensor proof compression.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_tensor_proof_matvec.m -o test_tensor_proof_matvec
// Run:
//   ./test_tensor_proof_matvec

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

// ---------------------------------------------------------------------------
// Configuration for tensor proof
// ---------------------------------------------------------------------------
typedef struct {
    int num_vars;
    int sqrt_n;
} TensorProofConfig;

static const TensorProofConfig kConfigs[] = {
    { .num_vars = 10, .sqrt_n = 32 },
    { .num_vars = 14, .sqrt_n = 128 },
    { .num_vars = 16, .sqrt_n = 256 },
    { .num_vars = 18, .sqrt_n = 512 },
    { .num_vars = 20, .sqrt_n = 1024 },
};

// ---------------------------------------------------------------------------
// Weight blob for matrix
// ---------------------------------------------------------------------------
static NSData *make_blob_matrix(int rows, int cols, float *data) {
    int ws = rows * cols * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < rows * cols; i++) {
        fp16[i] = (_Float16)data[i];
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ---------------------------------------------------------------------------
// Generate test matrix and vector
// ---------------------------------------------------------------------------
static void generate_test_data(int rows, int cols, float **matrix_out, float **vector_out) {
    float *matrix = malloc(rows * cols * sizeof(float));
    float *vector = malloc(cols * sizeof(float));

    // Use small values (1-20) to avoid fp16 overflow in conv1x1 accumulation
    // fp16 mantissa is ~10 bits, exact integers only up to 2048
    // With accumulation, values > ~20 can cause overflow
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (float)((i + j) % 20 + 1);  // Values 1-20
        }
    }

    for (int j = 0; j < cols; j++) {
        vector[j] = 1.0f;
    }

    *matrix_out = matrix;
    *vector_out = vector;
}

// ---------------------------------------------------------------------------
// CPU matvec
// ---------------------------------------------------------------------------
static void cpu_matvec(const float *matrix, const float *vec, float *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vec[j];
        }
        result[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Build ANE MatVec program
// ---------------------------------------------------------------------------
static NSString *build_matvec_mil(int rows, int cols, int seq) {
    NSString *wpath = @"@model_path/weights/M.bin";
    NSString *conv_body = orion_mil_linear("mv", "x16", cols, rows, seq, [wpath UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", cols, seq];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = mv_out)[name = string(\"out\")];\n", rows, seq];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", cols, seq]],
        @"y");
}

// ---------------------------------------------------------------------------
// Test MatVec at a specific configuration
// ---------------------------------------------------------------------------
static void test_matvec(int rows, int cols, int iterations) {
    printf("\n=== Tensor Proof MatVec: %dx%d ===\n", rows, cols);

    int seq = 16;
    printf("  Using batch seq=%d (ANE minimum)\n", seq);

    // Generate test data
    float *matrix, *vector;
    generate_test_data(rows, cols, &matrix, &vector);

    // CPU baseline
    float *cpu_result = malloc(rows * sizeof(float));
    cpu_matvec(matrix, vector, cpu_result, rows, cols);

    printf("  Expected (CPU) first 4: [%.1f, %.1f, %.1f, %.1f]\n",
           cpu_result[0], cpu_result[1], cpu_result[2], cpu_result[3]);

    // Build ANE program
    NSString *prog_text = build_matvec_mil(rows, cols, seq);
    NSData *blob = make_blob_matrix(rows, cols, matrix);
    NSDictionary *wdict = @{@"@model_path/weights/M.bin": @{@"offset": @0, @"data": blob}};

    clock_t compile_start = clock();
    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "tensor_matvec");
    clock_t compile_end = clock();
    double compile_ms = (double)(compile_end - compile_start) / CLOCKS_PER_SEC * 1000.0;

    if (!prog) {
        printf("  FAIL: program compile failed\n");
        g_fail++;
        free(matrix); free(vector); free(cpu_result);
        return;
    }
    printf("  Compile: %.2f ms\n", compile_ms);

    // Create IOSurfaces
    IOSurfaceRef ioX = orion_tensor_create_f32(cols, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(rows, seq);

    // Write input: x[c,s] = vector[c]
    IOSurfaceLock(ioX, 0, NULL);
    float *pX = (float *)IOSurfaceGetBaseAddress(ioX);
    for (int c = 0; c < cols; c++) {
        for (int s = 0; s < seq; s++) {
            pX[c * seq + s] = vector[c];
        }
    }
    IOSurfaceUnlock(ioX, 0, NULL);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    if (!ok) {
        printf("  FAIL: eval failed\n");
        g_fail++;
        CFRelease(ioX); CFRelease(ioY);
        orion_release_program(prog);
        free(matrix); free(vector); free(cpu_result);
        return;
    }
    printf("  PASS: eval succeeds\n");

    // Benchmark
    clock_t bench_start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t bench_end = clock();
    double total_ms = (double)(bench_end - bench_start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / iterations;

    // Read output: y[o,s] at pY[o*seq + s]
    float *ane_result = malloc(rows * sizeof(float));
    IOSurfaceLock(ioY, kIOSurfaceLockReadOnly, NULL);
    float *pY = (float *)IOSurfaceGetBaseAddress(ioY);

    for (int o = 0; o < rows; o++) {
        ane_result[o] = pY[o * seq + 0];
    }
    IOSurfaceUnlock(ioY, kIOSurfaceLockReadOnly, NULL);

    // Verify correctness
    float max_diff = 0;
    int max_idx = 0;
    for (int i = 0; i < rows; i++) {
        float diff = fabsf(ane_result[i] - cpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    printf("  Time: %.4f ms/iter (for %d parallel %dx%d MatVecs)\n", per_iter_ms, seq, rows, cols);
    printf("  Per MatVec: %.4f ms\n", per_iter_ms / seq);
    printf("  ANE result first 4: [%.1f, %.1f, %.1f, %.1f]\n",
           ane_result[0], ane_result[1], ane_result[2], ane_result[3]);
    printf("  Max diff: %.6f at index %d\n", max_diff, max_idx);

    float tolerance = rows > 256 ? 10.0f : 0.1f;
    if (max_diff < tolerance) {
        printf("  PASS: ANE matches CPU\n");
        g_pass++;
    } else {
        printf("  FAIL: ANE mismatch\n");
        g_fail++;
    }

    // Performance metrics
    double gflops = (2.0 * rows * cols * seq) / (per_iter_ms * 1e6);
    double ns_per_elem = (per_iter_ms * 1e6) / (rows * cols * seq);
    printf("  Batch GFLOPS: %.2f (implied)\n", gflops);
    printf("  ns/elem: %.2f\n", ns_per_elem);

    // Cleanup
    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
    free(matrix); free(vector); free(cpu_result); free(ane_result);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Tensor Proof MatVec on ANE ===\n");
        printf("Matrix-Vector multiply for Spartan/Spark tensor proof compression\n");
        printf("ANE sweet spot: dim >= 256, seq small\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        orion_mil_cache_clear();

        // Test various sizes
        int configs[] = {0, 1, 2, 3, 4};
        int iter_counts[] = {100, 50, 30, 20, 10};

        for (int i = 0; i < 1; i++) {  // Just second config
            int idx = 1;  // 128x128
            int sqrt_n = kConfigs[idx].sqrt_n;
            int iter = iter_counts[idx];
            test_matvec(sqrt_n, sqrt_n, iter);
        }

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        printf("\n=== Tensor Proof Hybrid Architecture ===\n");
        printf("\nFor Spartan/Spark tensor proof compression:\n");
        printf("  1. GPU computes tensor product t_L, t_R (Hadamard products)\n");
        printf("  2. ANE computes v = M * t_R (this test)\n");
        printf("  3. GPU runs sumcheck on v, t_L\n");
        printf("\nSplit at the MatVec boundary:\n");
        printf("  - zkMetal GPU: NTT, field arithmetic, hashing, transcript\n");
        printf("  - zkANE: Large dense MatVec (dim >= 256)\n");

        return g_fail > 0 ? 1 : 0;
    }
}