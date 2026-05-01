// test_capacity.m — ANE Capacity/Saturation Test
// Find the true ANE evaluation capacity by testing larger sizes.
//
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_capacity.m -o test_capacity
//   ./test_capacity

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

// Generate power-diagonal weight blob
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

static void write_fp32_to_iosurface(IOSurfaceRef s, float val, int count) {
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
}

// Test conv1x1 at various dim values with fixed seq=64
static void test_conv_scaling_dim(int dim, int seq, int iterations) {
    printf("\n=== Conv1x1 scaling: dim=%d, seq=%d, iter=%d ===\n", dim, seq, iterations);

    NSString *wpath = @"@model_path/weights/w.bin";
    NSString *conv_body = orion_mil_linear("c1", "x16", dim, dim, seq, [wpath UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = c1_out)[name = string(\"out\")];\n", dim, seq];

    NSString *prog_text = orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");

    NSData *blob = make_blob_identity(dim);
    NSString *wkey = [NSString stringWithFormat:@"@model_path/weights/w.bin"];
    NSDictionary *wdict = @{wkey: @{@"offset": @0, @"data": blob}};

    orion_mil_cache_clear();

    clock_t compile_start = clock();
    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "capacity");
    clock_t compile_end = clock();
    double compile_ms = (double)(compile_end - compile_start) / CLOCKS_PER_SEC * 1000.0;

    if (!prog) {
        printf("  FAIL: Program compilation failed\n");
        g_fail++;
        return;
    }
    printf("  Compile: %.2f ms\n", compile_ms);

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;
    double ns_per_elem = (ms * 1e6) / (dim * seq);

    printf("  Eval: %.4f ms/iter\n", ms);
    printf("  ns/elem: %.2f\n", ns_per_elem);
    printf("  Est throughput: %.0f Mops/s\n", 1000.0 / ms);

    CFRelease(ioX);
    CFRelease(ioY);
}

// Test conv1x1 at various seq values with fixed dim=256
static void test_conv_scaling_seq(int dim, int seq, int iterations) {
    printf("\n=== Conv1x1 scaling: dim=%d, seq=%d, iter=%d ===\n", dim, seq, iterations);

    NSString *wpath = @"@model_path/weights/w.bin";
    NSString *conv_body = orion_mil_linear("c1", "x16", dim, dim, seq, [wpath UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = c1_out)[name = string(\"out\")];\n", dim, seq];

    NSString *prog_text = orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");

    NSData *blob = make_blob_identity(dim);
    NSString *wkey = [NSString stringWithFormat:@"@model_path/weights/w.bin"];
    NSDictionary *wdict = @{wkey: @{@"offset": @0, @"data": blob}};

    orion_mil_cache_clear();

    clock_t compile_start = clock();
    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "capacity");
    clock_t compile_end = clock();
    double compile_ms = (double)(compile_end - compile_start) / CLOCKS_PER_SEC * 1000.0;

    if (!prog) {
        printf("  FAIL: Program compilation failed\n");
        g_fail++;
        return;
    }
    printf("  Compile: %.2f ms\n", compile_ms);

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;
    double ns_per_elem = (ms * 1e6) / (dim * seq);

    printf("  Eval: %.4f ms/iter\n", ms);
    printf("  ns/elem: %.2f\n", ns_per_elem);
    printf("  Est throughput: %.0f Mops/s\n", 1000.0 / ms);

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE Capacity Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        printf("==========================================================\n");
        printf("Part 1: Scaling with dim (seq=64 fixed)\n");
        printf("==========================================================\n");

        // Test dim scaling with fixed seq=64
        test_conv_scaling_dim(64, 64, 50);
        test_conv_scaling_dim(128, 64, 50);
        test_conv_scaling_dim(256, 64, 50);
        test_conv_scaling_dim(512, 64, 50);
        test_conv_scaling_dim(1024, 64, 50);
        test_conv_scaling_dim(2048, 64, 30);
        // Note: dim=4096 likely exceeds SRAM - skip for now

        printf("\n==========================================================\n");
        printf("Part 2: Scaling with seq (dim=256 fixed)\n");
        printf("==========================================================\n");

        // Test seq scaling with fixed dim=256
        test_conv_scaling_seq(256, 64, 50);
        test_conv_scaling_seq(256, 128, 50);
        test_conv_scaling_seq(256, 256, 50);
        test_conv_scaling_seq(256, 512, 50);
        test_conv_scaling_seq(256, 1024, 30);
        test_conv_scaling_seq(256, 2048, 20);

        printf("\n==========================================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("==========================================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}