// test_conv_pcs_bench.m — Conv-PCS Saturation Analysis
// Sweep dim and seq to find ANE saturation points for Conv-PCS
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_conv_pcs_bench.m -o test_conv_pcs_bench

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <mach/mach_time.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "orion_mil_cache.h"

static double mach_time_to_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t elapsed = end - start;
    return (double)elapsed * info.numer / info.denom / 1e6;
}

// Benchmark config
typedef struct {
    int dim;
    int seq;
    const char *tag;
} Config;

static Config gConfigs[] = {
    // Baseline: dim=8, seq=16 (current hardcoded in conv_pcs)
    {8, 16, "baseline"},

    // Sweep dim at seq=16
    {16, 16, "dim_sweep"},
    {32, 16, "dim_sweep"},
    {64, 16, "dim_sweep"},
    {128, 16, "dim_sweep"},
    {256, 16, "dim_sweep"},

    // Sweep seq at dim=64
    {64, 64, "seq_sweep"},
    {64, 256, "seq_sweep"},

    // Larger dims at moderate seq
    {512, 64, "large_dim"},
    {1024, 64, "large_dim"},
};

static int gNumConfigs = sizeof(gConfigs) / sizeof(gConfigs[0]);

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Conv-PCS Saturation Analysis ===\n\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        printf("%-8s %-6s %-12s %-10s %-10s\n",
               "dim", "seq", "eval_ms", "ns/elem", "status");
        printf("%-8s %-6s %-12s %-10s %-10s\n",
               "-----", "-----", "--------", "--------", "------");

        for (int i = 0; i < gNumConfigs; i++) {
            int dim = gConfigs[i].dim;
            int seq = gConfigs[i].seq;
            const char *tag = gConfigs[i].tag;

            // Build MIL program using orion_mil_linear pattern
            NSString *prefix = @"bench";
            NSString *wpath = @"@model_path/weights/K.bin";

            NSString *conv_body = orion_mil_linear([prefix UTF8String], "x16",
                                                  dim, 1, seq,
                                                  [wpath UTF8String], NULL);

            NSMutableString *body = [NSMutableString string];
            [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
            [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
            [body appendString:conv_body];
            [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
            [body appendFormat:@"        tensor<fp32, [1, 1, 1, %d]> y = cast(dtype = to32, x = %@_out)[name = string(\"out\")];\n", seq, prefix];

            NSString *mil_text = orion_mil_program(body,
                @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
                @"y");

            // Create weight blob [1, dim, 1, 1]
            int ws = 1 * dim * 2;
            int tot = 128 + ws;
            uint8_t *b = (uint8_t *)calloc(tot, 1);
            b[0] = 1; b[4] = 2;
            b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
            *(uint32_t *)(b + 72) = ws;
            *(uint32_t *)(b + 80) = 128;
            _Float16 *fp16 = (_Float16 *)(b + 128);
            for (int j = 0; j < dim; j++) {
                fp16[j] = (_Float16)((j % 5) - 2) * 0.5f;  // Small values [-1, 1]
            }
            NSData *blob = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
            NSDictionary *wdict = @{wpath: @{@"offset": @0, @"data": blob}};

            // Compile
            char prog_tag[64];
            snprintf(prog_tag, sizeof(prog_tag), "conv_pcs_%s_%d_%d", tag, dim, seq);

            uint64_t t0 = mach_absolute_time();
            OrionProgram *prog = orion_compile_mil([mil_text UTF8String], wdict, prog_tag);
            uint64_t t1 = mach_absolute_time();
            double compile_ms = mach_time_to_ms(t0, t1);

            if (!prog) {
                printf("%-8d %-6d %-12s %-10s %-10s\n", dim, seq, "N/A", "N/A", "COMPILE_FAIL");
                continue;
            }

            // Create input surface
            IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
            IOSurfaceRef ioY = orion_tensor_create_f32(1, seq);

            // Write input data
            IOSurfaceLock(ioX, 0, NULL);
            float *pX = (float *)IOSurfaceGetBaseAddress(ioX);
            for (int j = 0; j < dim * seq; j++) {
                pX[j] = ((j % 7) - 3) * 0.3f;
            }
            IOSurfaceUnlock(ioX, 0, NULL);

            // Evaluate (multiple iterations for stable timing)
            const int iterations = 10;

            // Warmup (first iteration may include cache setup overhead)
            orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);

            uint64_t eval_start = mach_absolute_time();
            for (int it = 0; it < iterations; it++) {
                orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);
            }
            uint64_t eval_end = mach_absolute_time();

            double total_ms = mach_time_to_ms(eval_start, eval_end);
            double per_iter_ms = total_ms / iterations;

            long long total_elem = (long long)dim * seq;
            double ns_per_elem = (per_iter_ms * 1e6) / total_elem;

            printf("%-8d %-6d %-12.4f %-10.2f %-10s\n",
                   dim, seq, per_iter_ms, ns_per_elem, "OK");

            CFRelease(ioX);
            CFRelease(ioY);

            // Clear cache between configs
            orion_mil_cache_clear();
        }

        printf("\n========================================\n");
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return 0;
    }
}