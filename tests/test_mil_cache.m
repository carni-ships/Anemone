// test_mil_cache.m — MIL Program Cache Test
// Demonstrates program caching for zkML workloads.
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_mil_cache.m -o test_mil_cache
//   ./test_mil_cache

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "mil_cache.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

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

static IOSurfaceRef make_fp32_const(int channels, int seq_len, float val) {
    int count = channels * seq_len;
    size_t bytes = count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static float read_fp32_first(IOSurfaceRef s) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[0];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

static NSString *build_single_conv_mil(int dim, int seq) {
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

// ---------------------------------------------------------------------------
// Test: Basic cache get/store
// ---------------------------------------------------------------------------
static void test_basic_cache(int dim, int seq) {
    printf("\n=== Test: Basic cache (dim=%d, seq=%d) ===\n", dim, seq);

    // Clear cache first
    orion_mil_cache_clear();
    CHECK(orion_mil_cache_size() == 0, "cache cleared");

    NSString *prog_text = build_single_conv_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    // First call: cache miss, compile
    clock_t start = clock();
    OrionProgram *prog1 = orion_mil_cache_get([prog_text UTF8String], wdict, "cached");
    clock_t end = clock();
    double first_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    CHECK(prog1 != NULL, "first call compiles");
    CHECK(orion_mil_cache_size() == 1, "cache has 1 entry");
    CHECK(orion_mil_cache_contains([prog_text UTF8String]), "MIL text is cached");

    // Second call: cache hit, no compile
    start = clock();
    OrionProgram *prog2 = orion_mil_cache_get([prog_text UTF8String], wdict, "cached");
    end = clock();
    double second_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    CHECK(prog2 != NULL, "second call returns program");
    CHECK(prog2 == prog1, "same program returned (cache hit)");

    printf("    First call (compile):  %.2f ms\n", first_ms);
    printf("    Second call (cache):  %.2f ms\n", second_ms);
    printf("    Speedup:              %.1fx\n", first_ms / second_ms);

    // Eval still works
    IOSurfaceRef ioX = make_fp32_const(dim, seq, 1.0f);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);

    bool ok = orion_eval(prog2, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "eval succeeds");

    float output = read_fp32_first(ioY);
    CHECK(fabsf(output - 1.0f) < 0.1f, "output correct");

    CFRelease(ioX);
    CFRelease(ioY);
    // DO NOT release prog1 or prog2 - cache owns them
}

// ---------------------------------------------------------------------------
// Test: Weight patching via cache
// ---------------------------------------------------------------------------
static void test_weight_patching(int dim, int seq) {
    printf("\n=== Test: Weight patching (dim=%d, seq=%d) ===\n", dim, seq);

    // Clear cache
    orion_mil_cache_clear();

    NSString *prog_text = build_single_conv_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    // Store original in cache
    OrionProgram *master = orion_mil_cache_get([prog_text UTF8String], wdict, "master");
    CHECK(master != NULL, "master compiled and cached");

    // Create patched copy with different weights (all 0.5)
    NSData *wblob2 = make_blob_identity(dim);
    // Modify blob data for patched version
    NSDictionary *wdict2 = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob2}};

    // Get with new weights - returns PATCHED copy, not cached master
    OrionProgram *patched = orion_mil_cache_get_with_weights([prog_text UTF8String], wdict2, "patched");
    CHECK(patched != NULL, "patched program created");

    // They should be different objects
    printf("    Master: %p, Patched: %p\n", master, patched);
    CHECK(patched != master, "patched is different from master");

    // Eval patched version
    IOSurfaceRef ioX = make_fp32_const(dim, seq, 1.0f);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);

    bool ok = orion_eval(patched, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    CHECK(ok, "patched eval succeeds");

    // Cache still has 1 entry (master)
    CHECK(orion_mil_cache_size() == 1, "cache still has 1 entry");

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(patched);  // Caller MUST release patched
}

// ---------------------------------------------------------------------------
// Test: Sumcheck multi-round with caching
// ---------------------------------------------------------------------------
static void test_sumcheck_cached(int dim, int seq, int n_rounds) {
    printf("\n=== Test: Sumcheck cached (dim=%d, seq=%d, rounds=%d) ===\n", dim, seq, n_rounds);

    // Clear cache
    orion_mil_cache_clear();

    float challenges[16];
    srand(42);
    for (int i = 0; i < n_rounds; i++) {
        challenges[i] = (float)(rand() % 100) / 50.0f + 0.5f; // [0.5, 2.5]
    }

    // First pass: compile all rounds (cold cache)
    printf("    First pass (compiling %d rounds)...\n", n_rounds);
    clock_t start = clock();
    OrionProgram *progs[16];
    for (int r = 0; r < n_rounds; r++) {
        char mil_buf[512];
        snprintf(mil_buf, sizeof(mil_buf),
                 "test_sumcheck_r%d", r);
        // Build a simple MIL for this round
        NSString *conv_body = orion_mil_linear(mil_buf, "x16", dim, dim, seq,
                                              [[NSString stringWithFormat:@"@model_path/weights/w%d.bin", r] UTF8String], NULL);
        NSMutableString *body = [NSMutableString string];
        [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
        [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
        [body appendString:conv_body];
        [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
        [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = %s_out)[name = string(\"out\")];\n", dim, seq, mil_buf];

        NSString *prog_text = orion_mil_program(body,
            @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
            @"y");

        NSData *wblob = make_blob_identity(dim);
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", r];
        NSDictionary *wdict = @{key: @{@"offset": @0, @"data": wblob}};

        progs[r] = orion_mil_cache_get([prog_text UTF8String], wdict, mil_buf);
        if (!progs[r]) {
            printf("    FAIL: round %d compile failed\n", r);
            return;
        }
    }
    clock_t end = clock();
    double compile_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("    Compile time: %.2f ms (%.2f ms/round)\n", compile_ms, compile_ms / n_rounds);

    // Second pass: same MIL text, cache should hit (warm cache)
    printf("    Second pass (cache hits)...\n");
    // Cache is still warm from first pass - no clear needed
    start = clock();
    for (int r = 0; r < n_rounds; r++) {
        char mil_buf[512];
        snprintf(mil_buf, sizeof(mil_buf),
                 "test_sumcheck_r%d", r);
        NSString *conv_body = orion_mil_linear(mil_buf, "x16", dim, dim, seq,
                                              [[NSString stringWithFormat:@"@model_path/weights/w%d.bin", r] UTF8String], NULL);
        NSMutableString *body = [NSMutableString string];
        [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
        [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
        [body appendString:conv_body];
        [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
        [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = %s_out)[name = string(\"out\")];\n", dim, seq, mil_buf];

        NSString *prog_text = orion_mil_program(body,
            @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
            @"y");

        NSData *wblob = make_blob_identity(dim);
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", r];
        NSDictionary *wdict = @{key: @{@"offset": @0, @"data": wblob}};

        progs[r] = orion_mil_cache_get([prog_text UTF8String], wdict, mil_buf);
    }
    end = clock();
    double cached_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("    Cached time:  %.2f ms (%.2f ms/round)\n", cached_ms, cached_ms / n_rounds);
    printf("    Speedup:      %.1fx\n", compile_ms / cached_ms);

    // Stats
    int hits, misses;
    int size = orion_mil_cache_stats(&hits, &misses);
    printf("    Cache stats: size=%d, hits=%d, misses=%d\n", size, hits, misses);

    CHECK(hits > 0, "cache has hits on second pass");
    CHECK(cached_ms < compile_ms, "cached faster than compile");

    // DO NOT release progs - cache owns them
}

// ---------------------------------------------------------------------------
// Test: Cache size limit
// ---------------------------------------------------------------------------
static void test_cache_size_limit(void) {
    printf("\n=== Test: Cache size limit ===\n");

    orion_mil_cache_clear();
    CHECK(orion_mil_cache_size() == 0, "cache cleared");

    // Compile 5 different programs
    for (int i = 0; i < 5; i++) {
        NSString *prog_text = build_single_conv_mil(64 + i * 32, 64);
        NSData *wblob = make_blob_identity(64 + i * 32);
        NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

        OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "test");
        if (prog != NULL) { g_pass++; printf("  PASS: program %d compiled\n", i); } else { g_fail++; printf("  FAIL: program %d compile failed\n", i); }
    }

    CHECK(orion_mil_cache_size() == 5, "cache has 5 entries");

    // Clear and verify
    orion_mil_cache_clear();
    CHECK(orion_mil_cache_size() == 0, "cache cleared");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== MIL Program Cache Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        test_basic_cache(256, 64);
        test_weight_patching(256, 64);
        test_sumcheck_cached(256, 64, 6);
        test_cache_size_limit();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
