// test_ane_baseline.m — ANE Baseline Performance Test
// Benchmarks: single conv1x1, chained conv1x1, varying seq_len
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     tests/test_ane_baseline.m -o test_ane_baseline
// Run:
//   ./test_ane_baseline

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"

#define CH 256

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// Weight blob helpers
// ---------------------------------------------------------------------------

// Constant weight blob: all elements = val
static NSData *make_blob_const(int rows, int cols, float val) {
    int ws = rows * cols * 2; // fp16
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < rows * cols; i++) {
        fp16[i] = (_Float16)val;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Identity weight blob: diagonal = 1, others = 0
static NSData *make_blob_identity(int dim) {
    int ws = dim * dim * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < dim; i++) {
        fp16[i * dim + i] = (_Float16)1.0f;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ---------------------------------------------------------------------------
// IOSurface helpers
// ---------------------------------------------------------------------------

// Create fp32 IOSurface with sequential values (1.0, 2.0, 3.0, ...)
static IOSurfaceRef make_fp32_surface(int channels, int seq_len) {
    int count = channels * seq_len;
    size_t bytes = count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    if (!s) return NULL;
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = (float)(i + 1);
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

// Read fp32 surface first element
static float read_fp32_first(IOSurfaceRef s) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[0];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// ---------------------------------------------------------------------------
// Build MIL for single conv1x1 using mil_builder helper
// ---------------------------------------------------------------------------
static NSString *build_single_conv_mil(int dim, int seq) {
    NSString *conv_body = orion_mil_linear("c1", "x16", dim, dim, seq,
                                          "@model_path/weights/w.bin", NULL);
    NSString *body = [NSString stringWithFormat:
        @"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        @"%@"
        @"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        @"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = c1_out)[name = string(\"out\")];\n",
        dim, seq, conv_body, dim, seq];
    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// ---------------------------------------------------------------------------
// Build MIL for N chained conv1x1 layers
// ---------------------------------------------------------------------------
static NSString *build_chained_conv_mil(int n_layers, int dim, int seq) {
    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];

    NSString *prev = @"x16";
    for (int i = 0; i < n_layers; i++) {
        NSString *conv_body = orion_mil_linear([[NSString stringWithFormat:@"L%d", i] UTF8String],
                                              [prev UTF8String], dim, dim, seq,
                                              [[NSString stringWithFormat:@"@model_path/weights/w%d.bin", i] UTF8String], NULL);
        [body appendString:conv_body];
        prev = [NSString stringWithFormat:@"L%d_out", i];
    }

    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> out = cast(dtype = to32, x = %@)[name = string(\"out\")];\n", dim, seq, prev];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"out");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_single_conv1x1(int seq_len) {
    printf("\n=== Test: Single conv1x1 (seq=%d) ===\n", seq_len);

    NSString *prog_text = build_single_conv_mil(CH, seq_len);
    NSData *wblob = make_blob_identity(CH);
    NSDictionary *wdict = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": wblob}};

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "single_conv");
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    IOSurfaceRef ioX = make_fp32_surface(CH, seq_len);
    IOSurfaceRef ioY = orion_tensor_create_f32(CH, seq_len);

    IOSurfaceRef ins[] = {ioX};
    IOSurfaceRef outs[] = {ioY};

    // Warmup
    bool warmup_ok = orion_eval(prog, ins, 1, outs, 1);
    if (!warmup_ok) { printf("    warmup FAILED\n"); }

    // Benchmark
    const int iterations = 20;
    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        clock_t start = clock();
        bool ok = orion_eval(prog, ins, 1, outs, 1);
        clock_t end = clock();
        if (!ok) { printf("    eval FAILED at iter %d\n", i); break; }
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    double avg_ms = (total_time / iterations) * 1000.0;
    int elements = CH * seq_len;
    double ns_per_elem = (avg_ms * 1e6) / elements;
    printf("    avg=%.3f ms  elements=%d  ns/elem=%.2f\n", avg_ms, elements, ns_per_elem);

    // Verify: with identity weight, output should equal input
    float first_in = read_fp32_first(ioX);
    float first_out = read_fp32_first(ioY);
    float diff = fabsf(first_out - first_in);
    printf("    first_in=%.2f first_out=%.2f diff=%.6f\n", first_in, first_out, diff);
    CHECK(diff < 0.1f, "output ≈ input (identity weight)");

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

static void test_chained_conv1x1(int n_layers, int seq_len) {
    printf("\n=== Test: %d Chained conv1x1 (seq=%d) ===\n", n_layers, seq_len);

    NSString *prog_text = build_chained_conv_mil(n_layers, CH, seq_len);
    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    for (int i = 0; i < n_layers; i++) {
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        wdict[key] = @{@"offset": @0, @"data": make_blob_identity(CH)};
    }

    char tag[32];
    snprintf(tag, sizeof(tag), "chained%d", n_layers);
    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, tag);
    CHECK(prog != NULL, "program compiles");

    if (!prog) return;

    IOSurfaceRef ioX = make_fp32_surface(CH, seq_len);
    IOSurfaceRef ioY = orion_tensor_create_f32(CH, seq_len);

    IOSurfaceRef ins[] = {ioX};
    IOSurfaceRef outs[] = {ioY};

    // Warmup
    orion_eval(prog, ins, 1, outs, 1);

    // Benchmark
    const int iterations = 20;
    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        clock_t start = clock();
        bool ok = orion_eval(prog, ins, 1, outs, 1);
        clock_t end = clock();
        if (!ok) { printf("    eval FAILED at iter %d\n", i); break; }
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    double avg_ms = (total_time / iterations) * 1000.0;
    printf("    avg=%.3f ms\n", avg_ms);

    // With identity weights, output should = input
    float first_in = read_fp32_first(ioX);
    float first_out = read_fp32_first(ioY);
    float diff = fabsf(first_out - first_in);
    printf("    first_in=%.2f first_out=%.2f diff=%.6f\n", first_in, first_out, diff);
    CHECK(diff < 0.1f, "output ≈ input (identity weights)");

    CFRelease(ioX);
    CFRelease(ioY);
    orion_release_program(prog);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE Baseline Performance Test ===\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n", orion_compile_count());

        // Single conv1x1 at various seq lengths
        test_single_conv1x1(64);
        test_single_conv1x1(128);
        test_single_conv1x1(256);
        test_single_conv1x1(512);

        // Chained convs
        test_chained_conv1x1(3, 64);
        test_chained_conv1x1(5, 64);
        test_chained_conv1x1(10, 64);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
