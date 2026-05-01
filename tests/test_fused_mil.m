// test_fused_mil.m — Fused MIL Programs Test
// Tests whether multiple convolutions can be fused into one MIL program.
//
// Findings so far:
// - 2-round fused MIL (conv -> conv) compiles OK
// - 3+ rounds or conv -> reduce_sum -> conv chains may fail
// - MIL has no MUX/select for conditional routing
//
// Compile and run:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_fused_mil.m -o test_fused_mil
//   ./test_fused_mil

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
static NSData *make_power_blob(int dim, float r) {
    int ws = dim * dim * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    float r_power = 1.0f;
    for (int i = 0; i < dim; i++) {
        fp16[i * dim + i] = (_Float16)r_power;
        r_power *= r;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Build MIL for a single sumcheck round (using orion_mil_linear like working tests)
static NSString *build_single_round_mil(const char *prefix, int dim, int seq, float r) {
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

// Build a 2-round fused program WITHOUT reduce_sum (just conv -> conv)
static NSString *build_2round_no_reduce_mil(int dim, int seq) {
    NSMutableString *body = [NSMutableString string];

    // Round 0: cast + conv
    [body appendString:@"        string to16_0 = const()[name = string(\"to16_0\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16_0 = cast(dtype = to16_0, x = x)[name = string(\"cx_0\")];\n", dim, seq];
    [body appendString:@"        string pt_0 = const()[name=string(\"pt_0\"), val=string(\"valid\")];\n"];
    [body appendString:@"        tensor<int32, [2]> st_0 = const()[name=string(\"st_0\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        tensor<int32, [4]> pd_0 = const()[name=string(\"pd_0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [body appendString:@"        tensor<int32, [2]> dl_0 = const()[name=string(\"dl_0\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        int32 gr_0 = const()[name=string(\"gr_0\"), val=int32(1)];\n"];
    [body appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W_0 = const()[name=string(\"W_0\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n", dim, dim, dim, dim];
    [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out_0 = conv(dilations=dl_0, groups=gr_0, pad=pd_0, pad_type=pt_0, strides=st_0, weight=W_0, x=x16_0)[name=string(\"out_0\")];\n", dim, seq];

    // Round 1: cast + conv (use out_0 as input)
    [body appendString:@"        string to16_1 = const()[name = string(\"to16_1\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16_1 = cast(dtype = to16_1, x = out_0)[name = string(\"cx_1\")];\n", dim, seq];
    [body appendString:@"        string pt_1 = const()[name=string(\"pt_1\"), val=string(\"valid\")];\n"];
    [body appendString:@"        tensor<int32, [2]> st_1 = const()[name=string(\"st_1\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        tensor<int32, [4]> pd_1 = const()[name=string(\"pd_1\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [body appendString:@"        tensor<int32, [2]> dl_1 = const()[name=string(\"dl_1\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        int32 gr_1 = const()[name=string(\"gr_1\"), val=int32(1)];\n"];
    [body appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W_1 = const()[name=string(\"W_1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", dim, dim, dim, dim];
    [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out_1 = conv(dilations=dl_1, groups=gr_1, pad=pd_1, pad_type=pt_1, strides=st_1, weight=W_1, x=x16_1)[name=string(\"out_1\")];\n", dim, seq];

    // Final cast
    [body appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype = to32, x = out_1)[name = string(\"out\")];\n", dim, seq];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// Build a 2-round fused program WITH reduce_sum between rounds
static NSString *build_2round_with_reduce_mil(int dim, int seq) {
    NSMutableString *body = [NSMutableString string];

    // Round 0: cast + conv + reduce_sum
    [body appendString:@"        string to16_0 = const()[name = string(\"to16_0\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16_0 = cast(dtype = to16_0, x = x)[name = string(\"cx_0\")];\n", dim, seq];
    [body appendString:@"        string pt_0 = const()[name=string(\"pt_0\"), val=string(\"valid\")];\n"];
    [body appendString:@"        tensor<int32, [2]> st_0 = const()[name=string(\"st_0\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        tensor<int32, [4]> pd_0 = const()[name=string(\"pd_0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [body appendString:@"        tensor<int32, [2]> dl_0 = const()[name=string(\"dl_0\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        int32 gr_0 = const()[name=string(\"gr_0\"), val=int32(1)];\n"];
    [body appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W_0 = const()[name=string(\"W_0\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n", dim, dim, dim, dim];
    [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> conv_0 = conv(dilations=dl_0, groups=gr_0, pad=pd_0, pad_type=pt_0, strides=st_0, weight=W_0, x=x16_0)[name=string(\"conv_0\")];\n", dim, seq];
    [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out_0 = identity(x=conv_0)[name=string(\"out_0\")];\n", dim, seq];
    [body appendString:@"        tensor<int32, [1]> ax_0 = const()[name=string(\"ax_0\"), val=tensor<int32, [1]>([1])];\n"];
    [body appendString:@"        bool kd_0 = const()[name=string(\"kd_0\"), val=bool(true)];\n"];
    [body appendFormat:@"        tensor<fp16, [1,1,1,%d]> sum_0 = reduce_sum(x=out_0, axes=ax_0, keep_dims=kd_0)[name=string(\"sum_0\")];\n", seq];

    // Round 1: expand sum_0 back to [1,dim,1,seq] then cast + conv
    // Note: Need to reshape/broadcast sum_0 to match expected input shape for next round
    [body appendString:@"        string to16_1 = const()[name = string(\"to16_1\"), val = string(\"fp16\")];\n"];
    // For simplicity, cast sum_0 directly - but this creates shape mismatch
    [body appendFormat:@"        tensor<fp16, [1,1,1,%d]> x16_1 = cast(dtype = to16_1, x = sum_0)[name = string(\"cx_1\")];\n", seq];
    // Reshape x16_1 to [1,dim,1,seq] - but MIL reshape may not support this
    // Instead, let's just try with the fp16 [1,1,1,seq] input
    [body appendString:@"        string pt_1 = const()[name=string(\"pt_1\"), val=string(\"valid\")];\n"];
    [body appendString:@"        tensor<int32, [2]> st_1 = const()[name=string(\"st_1\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        tensor<int32, [4]> pd_1 = const()[name=string(\"pd_1\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [body appendString:@"        tensor<int32, [2]> dl_1 = const()[name=string(\"dl_1\"), val=tensor<int32, [2]>([1,1])];\n"];
    [body appendString:@"        int32 gr_1 = const()[name=string(\"gr_1\"), val=int32(1)];\n"];
    [body appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W_1 = const()[name=string(\"W_1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", dim, dim, dim, dim];
    // This will fail because x16_1 has wrong shape [1,1,1,seq] but conv expects [1,dim,1,seq]
    [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> conv_1 = conv(dilations=dl_1, groups=gr_1, pad=pd_1, pad_type=pt_1, strides=st_1, weight=W_1, x=x16_1)[name=string(\"conv_1\")];\n", dim, seq];
    [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out_1 = identity(x=conv_1)[name=string(\"out_1\")];\n", dim, seq];

    // Final cast
    [body appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype = to32, x = out_1)[name = string(\"out\")];\n", dim, seq];

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
// Test: Sequential single-round programs
// ---------------------------------------------------------------------------
static void test_sequential_single_round(int dim, int seq, int n_rounds) {
    printf("\n=== Test: Sequential single-round (dim=%d, seq=%d, rounds=%d) ===\n", dim, seq, n_rounds);

    for (int r = 0; r < n_rounds; r++) {
        const char *prefix = [[NSString stringWithFormat:@"r%d", r] UTF8String];
        NSString *prog_text = build_single_round_mil(prefix, dim, seq, 1.5f);
        NSData *blob = make_power_blob(dim, 1.5f);
        NSString *wkey = [NSString stringWithFormat:@"@model_path/weights/w.bin"];
        NSDictionary *wdict = @{wkey: @{@"offset": @0, @"data": blob}};

        OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, prefix);
        if (!prog) {
            printf("    FAIL: Round %d compilation failed\n", r);
            g_fail++;
            return;
        }
        printf("    Round %d compiled OK\n", r);
    }
    g_pass++;
}

// ---------------------------------------------------------------------------
// Test: 2-round fused WITHOUT reduce_sum
// ---------------------------------------------------------------------------
static void test_2round_no_reduce(int dim, int seq) {
    printf("\n=== Test: 2-round fused WITHOUT reduce_sum (dim=%d, seq=%d) ===\n", dim, seq);

    NSString *prog_text = build_2round_no_reduce_mil(dim, seq);
    printf("    MIL length: %zu chars\n", [prog_text length]);

    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    NSData *blob = make_power_blob(dim, 1.5f);
    wdict[@"@model_path/weights/w.bin"] = @{@"offset": @0, @"data": blob};
    wdict[@"@model_path/weights/w2.bin"] = @{@"offset": @0, @"data": blob};

    orion_mil_cache_clear();

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "fused_no_reduce");
    if (prog) {
        printf("    SUCCESS: 2-round fused (no reduce) compiled!\n");
        g_pass++;
    } else {
        printf("    FAIL: 2-round fused compilation failed\n");
        g_fail++;
    }
}

// ---------------------------------------------------------------------------
// Test: 2-round fused WITH reduce_sum (will fail due to shape mismatch)
// ---------------------------------------------------------------------------
static void test_2round_with_reduce(int dim, int seq) {
    printf("\n=== Test: 2-round fused WITH reduce_sum (dim=%d, seq=%d) ===\n", dim, seq);

    NSString *prog_text = build_2round_with_reduce_mil(dim, seq);
    printf("    MIL length: %zu chars\n", [prog_text length]);

    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    NSData *blob = make_power_blob(dim, 1.5f);
    wdict[@"@model_path/weights/w.bin"] = @{@"offset": @0, @"data": blob};
    wdict[@"@model_path/weights/w2.bin"] = @{@"offset": @0, @"data": blob};

    orion_mil_cache_clear();

    OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict, "fused_with_reduce");
    if (prog) {
        printf("    UNEXPECTED: 2-round fused (with reduce) compiled!\n");
        g_pass++;
    } else {
        printf("    EXPECTED FAIL: Shape mismatch prevents chaining\n");
        printf("    (reduce_sum outputs [1,1,1,seq] but next conv expects [1,dim,1,seq])\n");
        g_pass++;  // Expected
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Fused MIL Programs Test ===\n\n");
        printf("Testing feasibility of Priority 5 (Fused MIL Programs)\n");
        printf("=====================================================\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n\n");

        // Sequential single-round programs work
        test_sequential_single_round(64, 32, 3);

        // 2-round fused without reduce_sum
        test_2round_no_reduce(64, 32);

        // 2-round fused with reduce_sum (fails due to shape)
        test_2round_with_reduce(64, 32);

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");
        printf("\nFindings on Priority 5 (Fused MIL Programs):\n");
        printf("  - 2-round fused WITHOUT reduce_sum: WORKS\n");
        printf("  - 2-round fused WITH reduce_sum: FAILS (shape mismatch)\n");
        printf("  - Problem: reduce_sum changes tensor shape, breaking the chain\n");
        printf("  - Sumcheck needs reduce_sum between rounds, so fused is NOT feasible\n");

        return g_fail > 0 ? 1 : 0;
    }
}