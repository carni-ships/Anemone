// Minimal test to check sumcheck chain compilation
#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
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

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Minimal Chain Test ===\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n");

        int dim = 64, seq = 64, rounds = 10;
        NSData *blob = make_blob_identity(dim);
        NSString *wkey = @"@model_path/weights/w.bin";
        NSDictionary *wdict = @{wkey: @{@"offset": @0, @"data": blob}};

        printf("Compiling %d round programs...\n", rounds);

        for (int r = 0; r < rounds; r++) {
            const char *prefix = [[NSString stringWithFormat:@"sc%d", r] UTF8String];
            NSString *prog_text = build_single_round_mil(prefix, dim, seq);

            OrionProgram *prog = orion_mil_cache_get([prog_text UTF8String], wdict,
                                                     [[NSString stringWithFormat:@"r%d", r] UTF8String]);

            if (!prog) {
                printf("FAIL: Round %d compile failed\n", r);
                return 1;
            }
            printf("  Round %d compiled OK\n", r);

            orion_release_program(prog);
        }

        printf("All %d rounds compiled and released.\n", rounds);
        printf("compile_count=%d\n", orion_compile_count());

        return 0;
    }
}