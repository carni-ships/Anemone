// test_blob_pool.m — Weight Blob Pool Test
// Demonstrates weight blob pooling for reducing allocation overhead.
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m core/orion_blob_pool.m \
//     tests/test_blob_pool.m -o test_blob_pool
//   ./test_blob_pool

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "orion_mil_cache.h"
#import "orion_blob_pool.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// Generate a weight blob with a power-diagonal matrix for sumcheck
static NSData *make_power_blob(int dim, float r) {
    int ws = dim * dim * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < dim; i++) {
        float val = powf(r, (float)i);
        fp16[i * dim + i] = (_Float16)val;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
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
// Test: Basic blob pool get/put
// ---------------------------------------------------------------------------
static void test_basic_pool(int dim) {
    printf("\n=== Test: Basic blob pool (dim=%d) ===\n", dim);

    orion_blob_pool_clear();
    int active, cached;
    orion_blob_pool_stats(&active, &cached);
    CHECK(active == 0, "pool cleared");

    // Create a blob
    NSData *blob1 = make_power_blob(dim, 1.5f);

    // Get from pool (first time - miss)
    NSData *pooled1 = orion_blob_pool_get(blob1.bytes, blob1.length);
    CHECK(pooled1 != NULL, "pool returns data");
    CHECK([pooled1 isEqualToData:blob1], "pooled data matches original");

    // Get same content again - should hit
    NSData *pooled2 = orion_blob_pool_get(blob1.bytes, blob1.length);
    CHECK(pooled2 != NULL, "pool returns data on second call");
    CHECK(pooled1 == pooled2, "same content returns same NSData (cache hit)");

    // Release and get again - should still hit
    orion_blob_pool_release(pooled2);
    NSData *pooled3 = orion_blob_pool_get(blob1.bytes, blob1.length);
    CHECK(pooled3 != NULL, "pool returns data after release");
    CHECK(pooled3 == pooled1, "same content after release returns same NSData");

    // Stats check
    orion_blob_pool_stats(&active, &cached);
    printf("    Pool stats: active=%d, cached=%d\n", active, cached);
    CHECK(active == 1, "pool has 1 entry");

    orion_blob_pool_release(pooled1);
    orion_blob_pool_release(pooled3);

    // Create different blob with different content
    NSData *blob2 = make_power_blob(dim, 2.0f);  // Different r

    NSData *pooled4 = orion_blob_pool_get(blob2.bytes, blob2.length);
    CHECK(pooled4 != pooled1, "different content returns different NSData");

    orion_blob_pool_stats(&active, &cached);
    printf("    Pool stats: active=%d, cached=%d\n", active, cached);
    CHECK(active == 2, "pool has 2 entries");

    orion_blob_pool_release(pooled4);
}

// ---------------------------------------------------------------------------
// Test: Pool integration with MIL cache
// ---------------------------------------------------------------------------
static void test_pool_with_mil_cache(int dim, int seq) {
    printf("\n=== Test: Pool with MIL cache (dim=%d, seq=%d) ===\n", dim, seq);

    orion_mil_cache_clear();
    orion_blob_pool_clear();

    NSString *prog_text = build_single_conv_mil(dim, seq);
    NSData *wblob = make_power_blob(dim, 1.5f);

    // Get pooled blob
    NSData *pooledBlob = orion_blob_pool_get(wblob.bytes, wblob.length);
    CHECK(pooledBlob != NULL, "got pooled blob");

    // Use pooled data in wdict
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": pooledBlob}};

    // First compile
    clock_t start = clock();
    OrionProgram *prog1 = orion_mil_cache_get([prog_text UTF8String], wdict, "pooled");
    clock_t end = clock();
    double compile_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    CHECK(prog1 != NULL, "first compile succeeds");

    // Second call with same weights - should hit MIL cache
    start = clock();
    OrionProgram *prog2 = orion_mil_cache_get([prog_text UTF8String], wdict, "pooled");
    end = clock();
    double cached_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    CHECK(prog2 == prog1, "cache hit returns same program");

    printf("    Compile: %.2f ms, Cached: %.2f ms\n", compile_ms, cached_ms);

    orion_blob_pool_release(pooledBlob);
}

// ---------------------------------------------------------------------------
// Test: Pool reuse across iterations
// ---------------------------------------------------------------------------
static void test_pool_reuse(int dim, int n_iters) {
    printf("\n=== Test: Pool reuse (dim=%d, iters=%d) ===\n", dim, n_iters);

    orion_blob_pool_clear();
    orion_mil_cache_clear();

    NSString *prog_text = build_single_conv_mil(dim, 64);
    NSData *sharedBlob = make_power_blob(dim, 1.5f);

    // Baseline: compile with fresh blobs each time
    clock_t start = clock();
    for (int i = 0; i < n_iters; i++) {
        NSData *blob = make_power_blob(dim, 1.5f);
        NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": blob}};
        OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "fresh");
        if (!prog && i == 0) { printf("  FAIL: first compile failed\n"); return; }
    }
    clock_t end = clock();
    double fresh_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    // With pooled blob
    NSData *pooledBlob = orion_blob_pool_get(sharedBlob.bytes, sharedBlob.length);
    orion_mil_cache_clear();  // Clear MIL cache between tests

    start = clock();
    for (int i = 0; i < n_iters; i++) {
        NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": pooledBlob}};
        OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "reuse");
    }
    end = clock();
    double pooled_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("    Fresh blobs: %.2f ms\n", fresh_ms);
    printf("    Pooled blob: %.2f ms\n", pooled_ms);

    orion_blob_pool_release(pooledBlob);
}

// ---------------------------------------------------------------------------
// Test: Different blobs pool correctly
// ---------------------------------------------------------------------------
static void test_different_blobs(void) {
    printf("\n=== Test: Different blobs pool correctly ===\n");

    orion_blob_pool_clear();

    NSData *blob1 = make_power_blob(64, 1.5f);
    NSData *blob2 = make_power_blob(64, 2.0f);
    NSData *blob3 = make_power_blob(128, 1.5f);  // Different dim

    NSData *p1 = orion_blob_pool_get(blob1.bytes, blob1.length);
    NSData *p2 = orion_blob_pool_get(blob2.bytes, blob2.length);
    NSData *p3 = orion_blob_pool_get(blob3.bytes, blob3.length);

    CHECK(p1 != p2, "different content -> different entries");
    CHECK(p1 != p3, "different dim -> different entries");
    CHECK(p2 != p3, "different r -> different entries");

    // Release all
    orion_blob_pool_release(p1);
    orion_blob_pool_release(p2);
    orion_blob_pool_release(p3);

    int active, cached;
    orion_blob_pool_stats(&active, &cached);
    CHECK(active == 3, "pool has 3 entries");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Weight Blob Pool Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        test_basic_pool(64);
        test_basic_pool(256);
        test_pool_with_mil_cache(256, 64);
        test_pool_reuse(256, 20);
        test_different_blobs();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}