// test_sram_limit.m — Test ANE SRAM Limit
// Find when 32MB SRAM limit causes perf degradation
//
// dim=4096: weights=32MB, activations too large → should fail or slow
//

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

static void test_dim(int dim, int seq, int iterations) {
    printf("\n=== dim=%d, seq=%d, iter=%d ===\n", dim, seq, iterations);

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
    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "sram_test");
    clock_t compile_end = clock();
    double compile_ms = (double)(compile_end - compile_start) / CLOCKS_PER_SEC * 1000.0;

    if (!prog) {
        printf("  FAIL: Program compilation failed (dim=%d may exceed ANE limits)\n", dim);
        printf("  Memory estimate: weights=%.1f MB\n", (dim * dim * 2) / (1024.0 * 1024.0));
        g_fail++;
        return;
    }
    printf("  Compile: %.2f ms\n", compile_ms);

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    if (!ok) {
        printf("  FAIL: Eval failed (possible SRAM exceeded)\n");
        g_fail++;
        CFRelease(ioX);
        CFRelease(ioY);
        return;
    }
    g_pass++;

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;
    double ns_per_elem = (ms * 1e6) / (dim * seq);
    double weight_mb = (dim * dim * 2) / (1024.0 * 1024.0);

    printf("  Eval: %.4f ms/iter\n", ms);
    printf("  ns/elem: %.2f\n", ns_per_elem);
    printf("  Weight memory: %.1f MB\n", weight_mb);
    printf("  Est throughput: %.0f Mops/s\n", 1000.0 / ms);

    CFRelease(ioX);
    CFRelease(ioY);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE SRAM Limit Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n");
        printf("Testing SRAM boundary (32MB on-chip)\n\n");

        // Test near and at SRAM limit
        // dim=2048: 2048^2 * 2 bytes = 8MB weights
        // dim=3072: 3072^2 * 2 bytes = 18MB weights
        // dim=4096: 4096^2 * 2 bytes = 32MB weights (exactly at limit)
        // dim=4480: 4480^2 * 2 bytes = 38MB weights (exceeds limit)

        test_dim(2048, 64, 30);
        test_dim(3072, 64, 20);
        test_dim(4096, 64, 15);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}