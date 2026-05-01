// test_conv_poly_commit.m — ConvPolyCommit on ANE
// Polynomial commitment using ANE conv1x1 + reduce_sum.
// Tensorizes polynomial P[1, log_deg, 1, 2^log_pts], chains
// rand_kernel conv1x1 + reduce_sum for log_deg folds.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     tests/test_conv_poly_commit.m -o test_conv_poly_commit
// Run:
//   ./test_conv_poly_commit

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

// ---------------------------------------------------------------------------
// Weight blob helpers
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

// Random weight blob (for commitment kernel)
static NSData *make_blob_random(int rows, int cols, unsigned int seed) {
    int ws = rows * cols * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        // Random FP16 in range [-1, 1]
        float f = (float)(rand() % 1000) / 500.0f - 1.0f;
        fp16[i] = (_Float16)f;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
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

static void read_fp32_range(IOSurfaceRef s, int start, int count, float *out) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) out[i] = p[start + i];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// ---------------------------------------------------------------------------
// Build polynomial commitment MIL
// commitment = reduce_sum(conv1x1(poly, rand_kernel), axis=log_deg)
// For now: just a single conv1x1 + reduce_sum for simplicity
// ---------------------------------------------------------------------------

// Build MIL for poly commitment with one conv1x1 + reduce_sum
static NSString *build_poly_commit_mil(int dim, int seq, const char *weight_path) {
    NSMutableString *m = [NSMutableString string];

    // Cast input to fp16
    [m appendString:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];

    // Conv op (using mil_builder helper)
    NSString *conv_body = orion_mil_linear("pc", "x16", dim, dim, seq, weight_path, NULL);
    [m appendString:conv_body];

    // reduce_sum along axis 1 (the channel dim)
    [m appendString:@"        tensor<int32, [1]> pc_ax = const()[name=string(\"pc_ax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool pc_kd = const()[name=string(\"pc_kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> pc_s = reduce_sum(x=pc_out, axes=pc_ax, keep_dims=pc_kd)[name=string(\"pc_sum\")];\n", seq];

    // Cast back to fp32
    [m appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,1,1,%d]> y = cast(dtype = to32, x = pc_s)[name = string(\"out\")];\n", seq];

    return orion_mil_program(m,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// ---------------------------------------------------------------------------
// Test: Simple conv1x1 + reduce_sum
// ---------------------------------------------------------------------------
static void test_simple_commit(int dim, int seq) {
    printf("\n=== Test: Simple conv1x1+reduce_sum (dim=%d, seq=%d) ===\n", dim, seq);

    NSString *prog_text = build_poly_commit_mil(dim, seq, "@model_path/weights/w.bin");
    NSData *wblob = make_blob_const(dim, dim, 1.0f);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "poly_commit");
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    // Input = constant 1.0 (smaller to avoid FP16 overflow)
    IOSurfaceRef ioX = make_fp32_const(dim, seq, 0.1f);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq); // reduced output

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        // With weight=1.0 and input=0.1:
        // conv output: each element = sum_j input[j] * weight[j,i] = dim * 0.1 * 1.0 = dim * 0.1 = 25.6
        // reduce_sum: sum over dim of (dim * 0.1) = dim * dim * 0.1 = 256 * 256 * 0.1 = 6553.6
        float expected = dim * dim * 0.1f;
        float actual = read_fp32_elem(ioY, 0);
        printf("    expected=%.2f actual=%.2f\n", expected, actual);
        CHECK(fabsf(actual - expected) < 5.0f, "output ≈ expected (within fp16 tolerance)");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: Polynomial tensor layout [1, log_deg, 1, 2^log_pts]
// Simulates P[deg][pt] -> reduce_sum over deg axis
// ---------------------------------------------------------------------------
static void test_poly_tensor(int log_pts) {
    printf("\n=== Test: Polynomial tensor (log_pts=%d) ===\n", log_pts);

    int seq = 1 << log_pts;  // 2^log_pts
    int dim = 8;             // 8 coefficients per point

    // Build commitment MIL
    NSString *prog_text = build_poly_commit_mil(dim, seq, "@model_path/weights/w.bin");
    NSData *wblob = make_blob_random(dim, dim, 42);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "poly_tensor");
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    // Input: P[deg][pt] = 1.0 for all
    IOSurfaceRef ioX = make_fp32_const(dim, seq, 1.0f);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        printf("    Commitment values (first 4 pts):\n");
        for (int pt = 0; pt < seq && pt < 4; pt++) {
            float v = read_fp32_elem(ioY, pt);
            printf("      pt[%d] = %.4f\n", pt, v);
        }
        // All outputs should be equal (same input pattern, same weight structure)
        float first = read_fp32_elem(ioY, 0);
        bool all_same = YES;
        for (int pt = 1; pt < seq; pt++) {
            float v = read_fp32_elem(ioY, pt);
            if (fabsf(v - first) > 0.01f) all_same = NO;
        }
        CHECK(all_same, "all commitment values equal (homogeneous input)");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: Chained commitment (multi-round)
// commitment = reduce_sum(conv1x1(conv1x1(..., rand_k2), rand_k1), axis)
// Simulates log_deg folds
// ---------------------------------------------------------------------------

static NSString *build_chained_commit_mil(int dim, int seq, int n_layers, const char **weight_paths) {
    NSMutableString *body = [NSMutableString string];

    [body appendString:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];

    NSString *prev = @"x16";
    for (int i = 0; i < n_layers; i++) {
        NSString *prefix = [NSString stringWithFormat:@"L%d", i];
        NSString *conv_body = orion_mil_linear([prefix UTF8String], [prev UTF8String], dim, dim, seq,
                                               weight_paths[i], NULL);
        [body appendString:conv_body];
        prev = [NSString stringWithFormat:@"L%d_out", i];
    }

    // Reduce sum
    [body appendString:@"        tensor<int32, [1]> pc_ax = const()[name=string(\"pc_ax\"), val=tensor<int32, [1]>([1])];\n"];
    [body appendString:@"        bool pc_kd = const()[name=string(\"pc_kd\"), val=bool(true)];\n"];
    [body appendFormat:@"        tensor<fp16, [1,1,1,%d]> pc_s = reduce_sum(x=%@, axes=pc_ax, keep_dims=pc_kd)[name=string(\"pc_sum\")];\n", seq, prev];
    [body appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1,1,1,%d]> y = cast(dtype = to32, x = pc_s)[name = string(\"out\")];\n", seq];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

static void test_chained_commit(int n_layers, int dim, int seq) {
    printf("\n=== Test: Chained commitment (%d layers, dim=%d, seq=%d) ===\n", n_layers, dim, seq);

    // Build weight paths
    const char *weights[10];
    NSMutableArray *weight_objs = [NSMutableArray array];
    for (int i = 0; i < n_layers; i++) {
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        weights[i] = [key UTF8String];
        [weight_objs addObject:key];
    }

    // Build MIL
    NSString *prog_text = build_chained_commit_mil(dim, seq, n_layers, weights);
    CHECK(prog_text != nil, "MIL generated");

    // Build weight dict
    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    for (int i = 0; i < n_layers; i++) {
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        wdict[key] = @{@"offset": @0, @"data": make_blob_random(dim, dim, 100 + i)};
    }

    // Compile
    char tag[32];
    snprintf(tag, sizeof(tag), "chained_commit_%d", n_layers);
    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, tag);
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    // Input = constant 0.01 (avoid FP16 overflow in deep chains)
    IOSurfaceRef ioX = make_fp32_const(dim, seq, 0.01f);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    if (ok) {
        float first = read_fp32_elem(ioY, 0);
        printf("    first_output=%.4f\n", first);
        CHECK(!isinf(first) && !isnan(first), "output is finite");
    }

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Test: Throughput benchmark
// ---------------------------------------------------------------------------
static void test_commit_throughput(int dim, int seq, int n_layers) {
    printf("\n=== Test: Commitment throughput (dim=%d, seq=%d, layers=%d) ===\n", dim, seq, n_layers);

    const char *weights[10];
    NSMutableArray *weight_objs = [NSMutableArray array];
    for (int i = 0; i < n_layers; i++) {
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        weights[i] = [key UTF8String];
        [weight_objs addObject:key];
    }

    NSString *prog_text = build_chained_commit_mil(dim, seq, n_layers, weights);

    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    for (int i = 0; i < n_layers; i++) {
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        wdict[key] = @{@"offset": @0, @"data": make_blob_random(dim, dim, 100 + i)};
    }

    clock_t start = clock();
    char tag[32];
    snprintf(tag, sizeof(tag), "bench_commit_%d", n_layers);
    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, tag);
    clock_t compile_end = clock();
    double compile_ms = (double)(compile_end - start) / CLOCKS_PER_SEC * 1000.0;

    if (!prog) { printf("  FAIL: compile\n"); return; }
    printf("  Compile: %.2f ms\n", compile_ms);

    IOSurfaceRef ioX = make_fp32_const(dim, seq, 1.0f);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    // Warmup
    orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);

    // Benchmark
    const int iterations = 20;
    clock_t bench_start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t bench_end = clock();
    double total_ms = (double)(bench_end - bench_start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / iterations;

    printf("  %d iterations: %.2f ms total, %.3f ms/iter\n", iterations, total_ms, per_iter_ms);
    printf("  ns/elem: %.2f\n", (per_iter_ms * 1e6) / (dim * seq));

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE ConvPolyCommit Test ===\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n", orion_compile_count());

        // Simple conv + reduce_sum
        test_simple_commit(256, 64);
        test_simple_commit(256, 128);

        // Polynomial tensor with varying sizes
        test_poly_tensor(4);  // 16 points
        test_poly_tensor(6);  // 64 points

        // Chained commitment (simulates log_deg folds)
        test_chained_commit(3, 256, 64);
        test_chained_commit(5, 256, 64);
        test_chained_commit(10, 256, 64);

        // Throughput benchmarks
        test_commit_throughput(256, 64, 3);
        test_commit_throughput(256, 64, 5);
        test_commit_throughput(256, 64, 10);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
