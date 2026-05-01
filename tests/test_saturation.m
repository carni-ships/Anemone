// test_saturation.m — Find True ANE Saturation Point
// Push to larger sizes to find when ANE becomes compute-bound
//
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_saturation.m -o test_saturation
//   ./test_saturation

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
        printf("=== ANE Saturation Point Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        // Format: dim, seq, iterations
        int tests[][3] = {
            // Large dim scaling (seq=64 fixed)
            {4096, 64, 15},
            {4096, 128, 10},
            {4096, 256, 8},
            {4096, 512, 5},
            {4096, 1024, 3},
            {4096, 2048, 2},

            // dim=2048 scaling
            {2048, 256, 20},
            {2048, 512, 15},
            {2048, 1024, 10},
            {2048, 2048, 5},

            // dim=3072 scaling
            {3072, 128, 10},
            {3072, 256, 8},
            {3072, 512, 5},

            // Very large seq at dim=256
            {256, 4096, 20},
            {256, 8192, 10},
            {256, 16384, 5},
        };

        int n_tests = sizeof(tests) / sizeof(tests[0]);

        printf("%-20s %8s %8s %10s %10s %12s\n", "Test", "dim", "seq", "eval_ms", "ns/elem", "weight_MB");
        printf("%-20s %8s %8s %10s %10s %12s\n", "----", "---", "---", "-------", "--------", "---------");

        double baseline = 0;
        for (int i = 0; i < n_tests; i++) {
            int dim = tests[i][0];
            int seq = tests[i][1];
            int iter = tests[i][2];

            double ms = benchmark_conv(dim, seq, iter);
            if (ms < 0) {
                printf("%-20s %8d %8d %10s\n", "FAILED", dim, seq, "compile_err");
                continue;
            }

            double ns_per_elem = (ms * 1e6) / (dim * seq);
            double weight_mb = (dim * dim * 2.0) / (1024 * 1024);

            if (i == 0) baseline = ms;

            printf("dim=%-5d seq=%-5d %8d %8d %10.4f %10.3f %12.1f\n",
                   dim, seq, dim, seq, ms, ns_per_elem, weight_mb);
        }

        printf("\n==========================================================\n");
        printf("Saturation Analysis\n");
        printf("==========================================================\n");
        printf("Key finding: If eval time grows proportionally with dim*seq,\n");
        printf("we've hit compute-bound regime. If eval time stays flat,\n");
        printf("we're still dispatch-bound.\n");

        return 0;
    }
}