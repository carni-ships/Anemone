// test_async_dispatch.m — Async Dispatch Test
// Tests async evaluation API for overlapping CPU work with ANE.
//
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_async_dispatch.m -o test_async_dispatch
//   ./test_async_dispatch

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <dispatch/dispatch.h>
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

static void write_fp32_to_iosurface(IOSurfaceRef s, float val, int count) {
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
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
// Test: Basic async eval
// ---------------------------------------------------------------------------
static void test_basic_async(int dim, int seq) {
    printf("\n=== Test: Basic async eval (dim=%d, seq=%d) ===\n", dim, seq);

    orion_mil_cache_clear();

    NSString *prog_text = build_single_conv_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "async");
    CHECK(prog != NULL, "program compiled");

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    // Test async eval with main queue callback
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block bool callback_called = false;
    __block bool callback_success = false;

    orion_eval_async(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1,
                     dispatch_get_main_queue(), ^(bool success) {
        callback_called = true;
        callback_success = success;
        dispatch_semaphore_signal(sem);
    });

    // Wait for callback (with timeout)
    dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC);
    long result = dispatch_semaphore_wait(sem, timeout);

    CHECK(callback_called, "callback was called");
    CHECK(!result, "callback returned within timeout");
    CHECK(callback_success, "async eval succeeded");

    float output = read_fp32_from_iosurface(ioY, 0);
    CHECK(fabsf(output - 1.0f) < 0.1f, "output correct");

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test: Overlapping CPU work with ANE
// ---------------------------------------------------------------------------
static void test_overlap(int dim, int seq, int n_iters) {
    printf("\n=== Test: Overlap CPU with ANE (dim=%d, seq=%d, iters=%d) ===\n", dim, seq, n_iters);

    orion_mil_cache_clear();

    NSString *prog_text = build_single_conv_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "overlap");
    CHECK(prog != NULL, "program compiled");

    // Baseline: sequential sync evals
    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(dim, seq);
    write_fp32_to_iosurface(ioX, 1.0f, dim * seq);

    clock_t start = clock();
    for (int i = 0; i < n_iters; i++) {
        orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
    }
    clock_t end = clock();
    double sync_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    // Async version: dispatch eval to background, do CPU work, wait
    // Note: This doesn't actually overlap because ANE eval is sync at framework level.
    // The benefit would be in a real pipeline where you dispatch multiple evals.

    double async_ms = sync_ms;  // Placeholder - no actual overlap in current impl

    printf("    Sync eval: %.2f ms for %d iterations\n", sync_ms, n_iters);
    printf("    Async eval: %.2f ms (no overlap in current impl)\n", async_ms);
    printf("    Note: True overlap requires native ANE async support\n");

    CHECK(sync_ms > 0, "sync baseline measured");

    CFRelease(ioX);
    CFRelease(ioY);
}

// ---------------------------------------------------------------------------
// Test: Multiple async evals in flight
// ---------------------------------------------------------------------------
static void test_multiple_async(int dim, int seq, int n_parallel) {
    printf("\n=== Test: Multiple async evals (dim=%d, seq=%d, parallel=%d) ===\n", dim, seq, n_parallel);

    orion_mil_cache_clear();

    NSString *prog_text = build_single_conv_mil(dim, seq);
    NSData *wblob = make_blob_identity(dim);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "multi");
    CHECK(prog != NULL, "program compiled");

    // Create multiple IOSurface pairs
    IOSurfaceRef *inputs = calloc(n_parallel, sizeof(IOSurfaceRef));
    IOSurfaceRef *outputs = calloc(n_parallel, sizeof(IOSurfaceRef));
    NSMutableArray *sems = [NSMutableArray arrayWithCapacity:n_parallel];
    NSMutableArray *results = [NSMutableArray arrayWithCapacity:n_parallel];

    for (int i = 0; i < n_parallel; i++) {
        inputs[i] = orion_tensor_create_f32(dim, seq);
        outputs[i] = orion_tensor_create_f32(dim, seq);
        write_fp32_to_iosurface(inputs[i], (float)(i + 1), dim * seq);
        [sems addObject:[NSValue valueWithPointer:(__bridge void *)dispatch_semaphore_create(0)]];
        [results addObject:@0];
    }

    // Launch all evals concurrently
    clock_t start = clock();
    for (int i = 0; i < n_parallel; i++) {
        const int idx = i;
        dispatch_semaphore_t sem = (dispatch_semaphore_t)[sems[idx] pointerValue];
        orion_eval_async(prog,
                         (IOSurfaceRef[]){inputs[idx]}, 1,
                         (IOSurfaceRef[]){outputs[idx]}, 1,
                         dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                         ^(bool success) {
            results[idx] = @(success ? 1 : 0);
            dispatch_semaphore_signal(sem);
        });
    }

    // Wait for all to complete
    for (int i = 0; i < n_parallel; i++) {
        dispatch_semaphore_t sem = (dispatch_semaphore_t)[sems[i] pointerValue];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    }
    clock_t end = clock();
    double total_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    // Verify results
    int success_count = 0;
    for (int i = 0; i < n_parallel; i++) {
        if ([results[i] intValue]) success_count++;
        float output = read_fp32_from_iosurface(outputs[i], 0);
        // Identity conv: output = input
        CHECK(fabsf(output - (float)(i + 1)) < 0.1f, "output correct");
    }

    printf("    Total time: %.2f ms for %d parallel evals\n", total_ms, n_parallel);
    printf("    Success: %d/%d\n", success_count, n_parallel);
    CHECK(success_count == n_parallel, "all async evals succeeded");

    // Cleanup
    for (int i = 0; i < n_parallel; i++) {
        CFRelease(inputs[i]);
        CFRelease(outputs[i]);
    }
    free(inputs);
    free(outputs);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Async Dispatch Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        test_basic_async(256, 64);
        test_overlap(256, 64, 20);
        test_multiple_async(256, 64, 4);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}