// test_sumcheck_chain_large.m — Large Sumcheck Chain Test
// Push sumcheck chains beyond 10 rounds to find compilation limits.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_sumcheck_chain_large.m -o test_sumcheck_chain_large
//   ./test_sumcheck_chain_large

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

// ---------------------------------------------------------------------------
// Weight blob (identity matrix)
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
    for (int i = 0; i < dim; i++) fp16[i * dim + i] = (_Float16)1.0f;
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ---------------------------------------------------------------------------
// Build single sumcheck round MIL
// ---------------------------------------------------------------------------
static NSString *build_single_round_mil(const char *prefix, int dim, int seq) {
    NSString *wpath = @"@model_path/weights/w.bin";
    NSString *conv_body = orion_mil_linear(prefix, "x16", dim, dim, seq, [wpath UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];

    [body appendFormat:@"        tensor<int32, [1]> %@_ax = const()[name=string(\"%@_ax\"), val=tensor<int32, [1]>([1])];\n", @(prefix), @(prefix)];
    [body appendFormat:@"        bool %@_kd = const()[name=string(\"%@_kd\"), val=bool(true)];\n", @(prefix), @(prefix)];
    [body appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_sum = reduce_sum(x=%@_out, axes=%@_ax, keep_dims=%@_kd)[name=string(\"%@_sum\")];\n",
     seq, @(prefix), @(prefix), @(prefix), @(prefix), @(prefix)];

    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1,1,1,%d]> y = cast(dtype = to32, x = %@_sum)[name = string(\"out\")];\n", seq, @(prefix)];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// ---------------------------------------------------------------------------
// Test: Sequential chain at various depths
// ---------------------------------------------------------------------------
static void test_sequential_chain(int dim, int seq, int n_rounds) {
    printf("\n=== Sequential Chain: dim=%d, seq=%d, rounds=%d ===\n", dim, seq, n_rounds);

    int compile_before = orion_compile_count();

    OrionProgram **progs = calloc(n_rounds, sizeof(OrionProgram*));
    NSData *blob = make_blob_identity(dim);
    NSString *wkey = @"@model_path/weights/w.bin";
    NSDictionary *wdict = @{wkey: @{@"offset": @0, @"data": blob}};

    // Compile all rounds
    for (int r = 0; r < n_rounds; r++) {
        const char *prefix = [[NSString stringWithFormat:@"sc%d", r] UTF8String];
        NSString *prog_text = build_single_round_mil(prefix, dim, seq);

        progs[r] = orion_mil_cache_get([prog_text UTF8String], wdict,
                                        [[NSString stringWithFormat:@"chain_r%d", r] UTF8String]);

        if (!progs[r]) {
            printf("  FAIL: Round %d compilation failed\n", r);
            g_fail++;
            free(progs);
            return;
        }
    }

    int compile_after = orion_compile_count();
    printf("  Compiles used: %d (before=%d, after=%d)\n",
           compile_after - compile_before, compile_before, compile_after);

    // Create IO surfaces
    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

    // Write input
    IOSurfaceLock(ioX, 0, NULL);
    float *pX = (float *)IOSurfaceGetBaseAddress(ioX);
    for (int i = 0; i < dim * seq; i++) pX[i] = 1.0f;
    IOSurfaceUnlock(ioX, 0, NULL);

    // Warmup
    for (int r = 0; r < n_rounds; r++) {
        orion_eval(progs[r], (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }

    // Benchmark
    const int iterations = 20;
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        for (int r = 0; r < n_rounds; r++) {
            orion_eval(progs[r], (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
        }
    }
    clock_t end = clock();
    double total_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double per_chain_ms = total_ms / iterations;
    double per_round_ms = per_chain_ms / n_rounds;

    printf("  Time: %.4f ms/chain, %.4f ms/round\n", per_chain_ms, per_round_ms);
    printf("  Throughput: %.0f chains/sec\n", 1000.0 / per_chain_ms);

    // Cleanup
    CFRelease(ioX);
    CFRelease(ioY);
    for (int r = 0; r < n_rounds; r++) {
        orion_release_program(progs[r]);
    }
    free(progs);

    g_pass++;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Large Sumcheck Chain Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        orion_mil_cache_clear();

        // Test chains at various depths
        test_sequential_chain(64, 64, 10);
        orion_mil_cache_clear();

        test_sequential_chain(64, 64, 50);
        orion_mil_cache_clear();

        test_sequential_chain(64, 64, 100);
        orion_mil_cache_clear();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}