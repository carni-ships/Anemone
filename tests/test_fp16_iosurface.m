// test_fp16_iosurface.m — Test fp16 IOSurface support
// Investigate whether ANE can use fp16 IOSurfaces directly without cast.
//
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_fp16_iosurface.m -o test_fp16_iosurface
//   ./test_fp16_iosurface

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "mil_cache.h"

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

// Build MIL with fp32 input + cast to fp16 (current approach)
static NSString *build_fp32_input_mil(int dim, int seq) {
    NSString *conv_body = orion_mil_linear("c1", "x16", dim, dim, seq,
                                          "@model_path/weights/w.bin", NULL);
    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = c1_out)[name = string(\"out\")];\n", dim, seq];
    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// Build MIL with fp16 input directly (potential optimization)
static NSString *build_fp16_input_mil(int dim, int seq) {
    NSString *conv_body = orion_mil_linear("c1", "x", dim, dim, seq,
                                          "@model_path/weights/w.bin", NULL);
    NSMutableString *body = [NSMutableString string];
    // No cast - input is already fp16
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = c1_out)[name = string(\"out\")];\n", dim, seq];
    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp16, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

static void write_fp32_to_iosurface(IOSurfaceRef s, float val, int count) {
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
}

static void write_fp16_to_iosurface(IOSurfaceRef s, float val, int count) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = (_Float16)val;
    IOSurfaceUnlock(s, 0, NULL);
}

static float read_fp32_from_iosurface(IOSurfaceRef s, int elem) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    float v = p[elem];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// ---------------------------------------------------------------------------
// Test: fp32 IOSurface with cast (baseline)
// ---------------------------------------------------------------------------
static void test_fp32_baseline(int dim, int seq) {
    printf("\n=== Test: fp32 IOSurface + cast (dim=%d, seq=%d) ===\n", dim, seq);

    orion_mil_cache_clear();

    NSString *prog_text = build_fp32_input_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "fp32");
    CHECK(prog != NULL, "fp32 program compiles");

    // Create fp32 input, fp32 output
    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "fp32 eval succeeds");

    // Benchmark
    const int iterations = 50;
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;

    float output = read_fp32_from_iosurface(ioY, 0);
    printf("    %.3f ms/iter, output[0]=%.4f\n", ms, output);

    // Identity conv1x1: output = input * 1.0 (at each position)
    // With input=1.0, output should be 1.0
    CHECK(fabsf(output - 1.0f) < 0.1f, "fp32 output correct (identity conv)");

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test: fp16 IOSurface with fp16 input (potential optimization)
// ---------------------------------------------------------------------------
static void test_fp16_direct(int dim, int seq) {
    printf("\n=== Test: fp16 IOSurface direct (dim=%d, seq=%d) ===\n", dim, seq);

    orion_mil_cache_clear();

    NSString *prog_text = build_fp16_input_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    printf("    MIL input type: fp16\n");
    printf("    MIL text:\n%s\n", [prog_text UTF8String]);

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "fp16");
    if (!prog) {
        printf("  FAIL: fp16 program failed to compile\n");
        g_fail++;
        return;
    }
    CHECK(prog != NULL, "fp16 program compiles");

    // Create fp16 input (using existing orion_tensor_create which is fp16)
    IOSurfaceRef ioX = orion_tensor_create(dim, seq);  // fp16
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);  // output still fp32
    write_fp16_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    if (!ok) {
        printf("  FAIL: fp16 eval failed\n");
        g_fail++;
        CFRelease(ioX);
        CFRelease(ioY);
        return;
    }
    CHECK(ok, "fp16 eval succeeds");

    // Benchmark
    const int iterations = 50;
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;

    float output = read_fp32_from_iosurface(ioY, 0);
    printf("    %.3f ms/iter, output[0]=%.4f\n", ms, output);

    CHECK(fabsf(output - 1.0f) < 0.1f, "fp16 direct output correct (identity conv)");

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test: Benchmark comparison
// ---------------------------------------------------------------------------
static void test_benchmark_comparison(int dim, int seq, int iterations) {
    printf("\n=== Benchmark: fp32+cast vs fp16 direct (dim=%d, seq=%d, iter=%d) ===\n", dim, seq, iterations);

    orion_mil_cache_clear();

    // fp32 version
    NSString *prog_fp32 = build_fp32_input_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};
    OrionProgram *prog32 = orion_mil_cache_get([prog_fp32 UTF8String], wdict, "fp32");
    IOSurfaceRef ioX32 = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY32 = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX32, 1.0f, dim * seq);
    orion_eval(prog32, (IOSurfaceRef[]){ioX32}, 1, (IOSurfaceRef[]){ioY32}, 1);  // warmup

    clock_t start32 = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog32, (IOSurfaceRef[]){ioX32}, 1, (IOSurfaceRef[]){ioY32}, 1);
    }
    clock_t end32 = clock();
    double ms32 = (double)(end32 - start32) / CLOCKS_PER_SEC * 1000.0 / iterations;

    // fp16 version
    orion_mil_cache_clear();
    NSString *prog_fp16 = build_fp16_input_mil(dim, seq);
    OrionProgram *prog16 = orion_mil_cache_get([prog_fp16 UTF8String], wdict, "fp16");
    IOSurfaceRef ioX16 = orion_tensor_create(dim, seq);
    IOSurfaceRef ioY16 = orion_tensor_create_f32(dim, seq);
    write_fp16_to_iosurface(ioX16, 1.0f, dim * seq);
    orion_eval(prog16, (IOSurfaceRef[]){ioX16}, 1, (IOSurfaceRef[]){ioY16}, 1);  // warmup

    clock_t start16 = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog16, (IOSurfaceRef[]){ioX16}, 1, (IOSurfaceRef[]){ioY16}, 1);
    }
    clock_t end16 = clock();
    double ms16 = (double)(end16 - start16) / CLOCKS_PER_SEC * 1000.0 / iterations;

    printf("    fp32+cast: %.3f ms/iter\n", ms32);
    printf("    fp16 direct: %.3f ms/iter\n", ms16);
    if (ms16 < ms32) {
        printf("    fp16 speedup: %.2fx\n", ms32 / ms16);
    } else {
        printf("    fp32+cast is faster by %.2fx\n", ms16 / ms32);
    }

    // Verify correctness - identity conv should give output = input = 1.0
    float out32 = read_fp32_from_iosurface(ioY32, 0);
    float out16 = read_fp32_from_iosurface(ioY16, 0);
    CHECK(fabsf(out32 - 1.0f) < 0.1f, "fp32 output correct");
    CHECK(fabsf(out16 - 1.0f) < 0.1f, "fp16 output correct");
    CHECK(fabsf(out32 - out16) < 0.01f, "outputs match");

    CFRelease(ioX32);
    CFRelease(ioY32);
    CFRelease(ioX16);
    CFRelease(ioY16);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== fp16 IOSurface Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        // Test small dims first
        test_fp32_baseline(64, 32);
        test_fp16_direct(64, 32);

        // Test larger
        test_fp32_baseline(256, 64);
        test_fp16_direct(256, 64);

        // Benchmark
        test_benchmark_comparison(256, 64, 50);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}