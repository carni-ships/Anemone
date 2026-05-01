// test_saturation_v2.m — Find True ANE Saturation Point
// More targeted tests to understand scaling behavior
//
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_saturation_v2.m -o test_saturation_v2
//   ./test_saturation_v2

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "orion_mil_cache.h"

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

static double benchmark_conv(int dim, int seq, int iterations) {
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

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "sat");
    if (!prog) return -1;

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Warmup
    orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0 / iterations;

    CFRelease(ioX);
    CFRelease(ioY);
    return ms;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE Saturation Point Test v2 ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        printf("==========================================================\n");
        printf("Test 1: Dimension scaling at SMALL seq (seq=16)\n");
        printf("(Isolates weight-dependent compute bound)\n");
        printf("==========================================================\n");
        printf("%-15s %8s %8s %10s %10s %10s\n", "Test", "dim", "seq", "eval_ms", "ns/elem", "wt_MB");
        printf("%-15s %8s %8s %10s %10s %10s\n", "----", "---", "---", "-------", "--------", "-----");

        int small_seq_tests[][3] = {
            {64, 16, 50},
            {128, 16, 50},
            {256, 16, 50},
            {512, 16, 50},
            {1024, 16, 50},
            {2048, 16, 50},
            {4096, 16, 30},
        };

        for (int i = 0; i < sizeof(small_seq_tests)/sizeof(small_seq_tests[0]); i++) {
            int dim = small_seq_tests[i][0];
            int seq = small_seq_tests[i][1];
            int iter = small_seq_tests[i][2];
            double ms = benchmark_conv(dim, seq, iter);
            if (ms < 0) { printf("dim=%d FAILED\n", dim); continue; }
            double ns = (ms * 1e6) / (dim * seq);
            double wt = (dim * dim * 2.0) / (1024*1024);
            printf("dim=%-5d       %8d %8d %10.4f %10.2f %10.1f\n", dim, dim, seq, ms, ns, wt);
        }

        printf("\n==========================================================\n");
        printf("Test 2: Dimension scaling at LARGE seq (seq=2048)\n");
        printf("(Tests memory bandwidth bound regime)\n");
        printf("==========================================================\n");
        printf("%-15s %8s %8s %10s %10s %10s\n", "Test", "dim", "seq", "eval_ms", "ns/elem", "wt_MB");
        printf("%-15s %8s %8s %10s %10s %10s\n", "----", "---", "---", "-------", "--------", "-----");

        int large_seq_tests[][3] = {
            {256, 2048, 30},
            {512, 2048, 30},
            {1024, 2048, 20},
            {2048, 2048, 15},
            {3072, 2048, 10},
            {4096, 2048, 8},
        };

        for (int i = 0; i < sizeof(large_seq_tests)/sizeof(large_seq_tests[0]); i++) {
            int dim = large_seq_tests[i][0];
            int seq = large_seq_tests[i][1];
            int iter = large_seq_tests[i][2];
            double ms = benchmark_conv(dim, seq, iter);
            if (ms < 0) { printf("dim=%d FAILED\n", dim); continue; }
            double ns = (ms * 1e6) / (dim * seq);
            double wt = (dim * dim * 2.0) / (1024*1024);
            printf("dim=%-5d       %8d %8d %10.4f %10.2f %10.1f\n", dim, dim, seq, ms, ns, wt);
        }

        printf("\n==========================================================\n");
        printf("Test 3: Pure memory bound test (small dim, huge seq)\n");
        printf("==========================================================\n");
        printf("%-15s %8s %8s %10s %10s %10s\n", "Test", "dim", "seq", "eval_ms", "ns/elem", "wt_MB");
        printf("%-15s %8s %8s %10s %10s %10s\n", "----", "---", "---", "-------", "--------", "-----");

        int mem_tests[][3] = {
            {32, 16384, 30},
            {64, 16384, 30},
            {128, 16384, 30},
            {256, 16384, 20},
            {512, 16384, 15},
        };

        for (int i = 0; i < sizeof(mem_tests)/sizeof(mem_tests[0]); i++) {
            int dim = mem_tests[i][0];
            int seq = mem_tests[i][1];
            int iter = mem_tests[i][2];
            double ms = benchmark_conv(dim, seq, iter);
            if (ms < 0) { printf("dim=%d FAILED\n", dim); continue; }
            double ns = (ms * 1e6) / (dim * seq);
            double wt = (dim * dim * 2.0) / (1024*1024);
            printf("dim=%-5d       %8d %8d %10.4f %10.2f %10.1f\n", dim, dim, seq, ms, ns, wt);
        }

        printf("\n==========================================================\n");
        printf("ANALYSIS: What limits performance at each regime?\n");
        printf("==========================================================\n");
        printf("1. Small dim (32-256), large seq: MEMORY-BOUND\n");
        printf("   - Time scales with seq (data movement)\n");
        printf("   - ns/elem improves as seq grows (better bandwidth util)\n");
        printf("\n");
        printf("2. Large dim (1024+), small seq: COMPUTE-BOUND\n");
        printf("   - Time scales with dim^2 (weight matrix multiply)\n");
        printf("   - Each output needs dim MACs\n");
        printf("\n");
        printf("3. Large dim + large seq: MIXED\n");
        printf("   - Both weight compute AND activation memory matter\n");

        return 0;
    }
}