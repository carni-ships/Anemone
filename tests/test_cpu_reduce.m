// test_cpu_reduce.m — CPU Reduce Fallback Test
// Compare ANE reduce_sum vs CPU sum for scalar reductions.
//
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_cpu_reduce.m -o test_cpu_reduce
//   ./test_cpu_reduce

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "orion_mil_cache.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// Generate identity weight blob
static NSData *make_blob_identity(int dim) {
    int ws = dim * dim * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < dim; i++) fp16[i * dim + i] = (_Float16)1.0f;
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// MIL with ANE reduce_sum (current approach)
static NSString *build_sumcheck_with_reduce(int dim, int seq) {
    NSString *conv_body = orion_mil_linear("c1", "x16", dim, dim, seq,
                                          "@model_path/weights/w.bin", NULL);
    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];

    // ANE reduce_sum along axis 1
    [body appendFormat:@"        tensor<int32, [1]> c1_ax = const()[name=string(\"c1_ax\"), val=tensor<int32, [1]>([1])];\n"];
    [body appendFormat:@"        bool c1_kd = const()[name=string(\"c1_kd\"), val=bool(true)];\n"];
    [body appendFormat:@"        tensor<fp16, [1,1,1,%d]> c1_sum = reduce_sum(x=c1_out, axes=c1_ax, keep_dims=c1_kd)[name=string(\"c1_sum\")];\n", seq];

    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1,1,1,%d]> y = cast(dtype = to32, x = c1_sum)[name = string(\"out\")];\n", seq];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// MIL without reduce_sum - CPU will do the sum
static NSString *build_sumcheck_no_reduce(int dim, int seq) {
    NSString *conv_body = orion_mil_linear("c1", "x16", dim, dim, seq,
                                          "@model_path/weights/w.bin", NULL);
    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];

    // No reduce_sum - output full conv result for CPU to sum
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = c1_out)[name = string(\"out\")];\n", dim, seq];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

static void write_fp32_to_iosurface(IOSurfaceRef s, float val, int count) {
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
}

// CPU sum of fp32 data from an fp32 IOSurface
static float cpu_sum_fp32(IOSurfaceRef s, int count) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    float sum = 0;
    for (int i = 0; i < count; i++) {
        sum += p[i];
    }
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return sum;
}

// ---------------------------------------------------------------------------
// Test: ANE reduce_sum benchmark
// ---------------------------------------------------------------------------
static void test_ane_reduce(int dim, int seq, int iterations) {
    printf("\n=== Test: ANE reduce_sum (dim=%d, seq=%d, iter=%d) ===\n", dim, seq, iterations);

    orion_mil_cache_clear();

    NSString *prog_text = build_sumcheck_with_reduce(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "ane_reduce");
    CHECK(prog != NULL, "ANE reduce program compiles");

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(seq, 1);  // [1,1,1,seq] output
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "ANE reduce eval succeeds");

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;

    // Read output - sum over seq dimension
    float output = cpu_sum_fp32(ioY, seq);
    printf("    %.3f ms/iter, output sum = %.4f\n", ms, output);
    CHECK(fabsf(output - (float)(dim * seq)) < 1.0f, "output correct");

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test: CPU reduce fallback benchmark
// ---------------------------------------------------------------------------
static void test_cpu_reduce_fallback(int dim, int seq, int iterations) {
    printf("\n=== Test: CPU reduce fallback (dim=%d, seq=%d, iter=%d) ===\n", dim, seq, iterations);

    orion_mil_cache_clear();

    NSString *prog_text = build_sumcheck_no_reduce(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "cpu_reduce");
    CHECK(prog != NULL, "CPU reduce program compiles");

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);  // Full [1,dim,1,seq] output
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "CPU reduce eval succeeds");

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
        // CPU sum of output tensor
        cpu_sum_fp32(ioY, dim * seq);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;

    float output = cpu_sum_fp32(ioY, dim * seq);
    printf("    %.3f ms/iter (ANE + CPU sum), output = %.4f\n", ms, output);
    CHECK(fabsf(output - (float)(dim * seq)) < 1.0f, "output correct");

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test: CPU sum overhead benchmark
// ---------------------------------------------------------------------------
static void test_cpu_sum_overhead(int dim, int seq, int iterations) {
    printf("\n=== Test: CPU sum overhead (dim=%d, seq=%d, iter=%d) ===\n", dim, seq, iterations);

    // Create an IOSurface with data
    IOSurfaceRef ioY = orion_tensor_create(dim, seq);  // fp16
    IOSurfaceLock(ioY, 0, NULL);
    _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(ioY);
    for (int i = 0; i < dim * seq; i++) p[i] = (_Float16)1.0f;
    IOSurfaceUnlock(ioY, 0, NULL);

    // Benchmark CPU sum (as fp16)
    clock_t start = clock();
    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        IOSurfaceLock(ioY, kIOSurfaceLockReadOnly, NULL);
        _Float16 *arr = (_Float16 *)IOSurfaceGetBaseAddress(ioY);
        for (int j = 0; j < dim * seq; j++) sum += (float)arr[j];
        IOSurfaceUnlock(ioY, kIOSurfaceLockReadOnly, NULL);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;

    int count = dim * seq;
    printf("    CPU sum: %.4f ms/iter for %d elements\n", ms, count);
    printf("    Per-element: %.2f ns\n", (ms * 1e6) / count);

    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test: Compare correct outputs
// ---------------------------------------------------------------------------
static void test_correctness(int dim, int seq) {
    printf("\n=== Test: Correctness (dim=%d, seq=%d) ===\n", dim, seq);

    orion_mil_cache_clear();

    // ANE reduce
    NSString *prog_reduce = build_sumcheck_with_reduce(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};
    OrionProgram *prog1 = orion_mil_cache_get([prog_reduce UTF8String], wdict, "reduce");

    // CPU reduce
    orion_mil_cache_clear();
    NSString *prog_cpu = build_sumcheck_no_reduce(dim, seq);
    OrionProgram *prog2 = orion_mil_cache_get([prog_cpu UTF8String], wdict, "cpu");

    // Create tensors
    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // ANE reduce output [1,1,1,seq]
    IOSurfaceRef ioY1 = orion_tensor_create_f32(seq, 1);
    // CPU reduce output [1,dim,1,seq]
    IOSurfaceRef ioY2 = orion_tensor_create_f32(dim, seq);

    // Evaluate ANE reduce
    bool ok1 = orion_eval(prog1, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY1}, 1);
    CHECK(ok1, "ANE reduce eval succeeds");
    float sum1 = cpu_sum_fp32(ioY1, seq);

    // Evaluate CPU reduce
    bool ok2 = orion_eval(prog2, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY2}, 1);
    CHECK(ok2, "CPU reduce eval succeeds");
    float sum2 = cpu_sum_fp32(ioY2, dim * seq);

    printf("    ANE reduce sum: %.4f (expected %.0f)\n", sum1, (float)(dim * seq));
    printf("    CPU reduce sum: %.4f (expected %.0f)\n", sum2, (float)(dim * seq));
    printf("    Diff: %.6f\n", fabsf(sum1 - sum2));

    CHECK(fabsf(sum1 - sum2) < 0.01f, "outputs match");
    CHECK(fabsf(sum1 - (float)(dim * seq)) < 1.0f, "ANE reduce correct");

    CFRelease(ioX);
    CFRelease(ioY1);
    CFRelease(ioY2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== CPU Reduce Fallback Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        // CPU sum overhead
        test_cpu_sum_overhead(256, 64, 1000);

        // Correctness
        test_correctness(256, 64);

        // Benchmarks
        test_ane_reduce(256, 64, 50);
        test_cpu_reduce_fallback(256, 64, 50);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}