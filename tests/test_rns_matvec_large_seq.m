// test_rns_matvec_large_seq.m — RNS-MatVec at Large Sequence Lengths
// Push RNS-MatVec beyond 256 seq to find scaling limits
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m \
//     tests/test_rns_matvec_large_seq.m -o test_rns_matvec_large_seq
// Run:
//   ./test_rns_matvec_large_seq

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <time.h>
#import <math.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "orion_mil_cache.h"

#define CH 256
#define N_RESIDUES 5

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// RNS moduli
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t mod;
    const char *name;
} RNSMod;

static const RNSMod kRNSMod[N_RESIDUES] = {
    { 3, "q0" }, { 5, "q1" }, { 7, "q2" }, { 11, "q3" }, { 13, "q4" },
};

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
// IOSurface helpers
// ---------------------------------------------------------------------------
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

static float read_fp32_elem(IOSurfaceRef s, int idx) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[idx];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// ---------------------------------------------------------------------------
// Build RNS single residue MIL program
// ---------------------------------------------------------------------------
static NSString *build_rns_single_mil(int dim, int seq, int residue_idx) {
    NSString *prefix = [NSString stringWithFormat:@"r%d", residue_idx];
    NSString *weight_path = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", residue_idx];

    NSString *conv_body = orion_mil_linear([prefix UTF8String], "x16", dim, dim, seq,
                                          [weight_path UTF8String], NULL);
    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = %@_out)[name = string(\"out\")];\n", dim, seq, prefix];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");
}

// ---------------------------------------------------------------------------
// Benchmark RNS-MatVec at given seq length
// ---------------------------------------------------------------------------
static double benchmark_rns_matvec(int seq_len, int iterations) {
    printf("\n=== RNS-MatVec seq=%d (iter=%d) ===\n", seq_len, iterations);

    OrionProgram *progs[N_RESIDUES];
    IOSurfaceRef ioXs[N_RESIDUES];
    IOSurfaceRef ioYs[N_RESIDUES];

    // Create programs
    clock_t compile_start = clock();
    for (int r = 0; r < N_RESIDUES; r++) {
        NSString *prog_text = build_rns_single_mil(CH, seq_len, r);
        NSString *key = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", r];
        NSData *blob = make_blob_identity(CH);
        NSDictionary *wdict = @{key: @{@"offset": @0, @"data": blob}};

        NSString *tag = [NSString stringWithFormat:@"rns_l%d", r];
        progs[r] = orion_mil_cache_get([prog_text UTF8String], wdict, [tag UTF8String]);
        if (!progs[r]) {
            printf("  FAIL: compile failed for residue %d\n", r);
            return -1;
        }
    }
    clock_t compile_end = clock();
    double compile_ms = (double)(compile_end - compile_start) / CLOCKS_PER_SEC * 1000.0;
    printf("  Compile: %.2f ms total (%.2f ms/residue)\n", compile_ms, compile_ms / N_RESIDUES);

    // Create surfaces
    for (int r = 0; r < N_RESIDUES; r++) {
        ioXs[r] = make_fp32_surface(CH, seq_len);
        ioYs[r] = orion_tensor_create_f32(CH, seq_len);
        if (!ioXs[r] || !ioYs[r]) {
            printf("  FAIL: IOSurface creation failed for residue %d\n", r);
            return -1;
        }
    }

    // Warmup
    for (int r = 0; r < N_RESIDUES; r++) {
        if (!orion_eval(progs[r], (IOSurfaceRef[]){ioXs[r]}, 1, (IOSurfaceRef[]){ioYs[r]}, 1)) {
            printf("  FAIL: warmup eval failed for residue %d\n", r);
            return -1;
        }
    }

    // Benchmark
    clock_t bench_start = clock();
    for (int i = 0; i < iterations; i++) {
        for (int r = 0; r < N_RESIDUES; r++) {
            orion_eval(progs[r], (IOSurfaceRef[]){ioXs[r]}, 1, (IOSurfaceRef[]){ioYs[r]}, 1);
        }
    }
    clock_t bench_end = clock();
    double total_ms = (double)(bench_end - bench_start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / iterations;
    double per_residue_ms = per_iter_ms / N_RESIDUES;
    int total_elements = N_RESIDUES * CH * seq_len;
    double ns_per_elem = (per_iter_ms * 1e6) / total_elements;
    double gflops = (N_RESIDUES * CH * CH * (double)seq_len * 2) / (per_iter_ms * 1e6);

    printf("  Time: %.4f ms/iter (%.4f ms/residue)\n", per_iter_ms, per_residue_ms);
    printf("  Elements: %d, ns/elem: %.2f\n", total_elements, ns_per_elem);
    printf("  GFLOPS: %.0f (implied)\n", gflops);

    // Verify correctness (identity weight → output ≈ input)
    float first_in = read_fp32_elem(ioXs[0], 0);
    float first_out = read_fp32_elem(ioYs[0], 0);
    float diff = fabsf(first_out - first_in);
    printf("  Correctness: first_in=%.2f first_out=%.2f diff=%.6f %s\n",
           first_in, first_out, diff, diff < 0.1f ? "PASS" : "FAIL");

    // Cleanup
    for (int r = 0; r < N_RESIDUES; r++) {
        CFRelease(ioXs[r]);
        CFRelease(ioYs[r]);
    }

    return per_iter_ms;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== RNS-MatVec Large Seq Test ===\n");
        printf("dim=%d residues=%d\n\n", CH, N_RESIDUES);

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        // Clear cache for consistent timing
        orion_mil_cache_clear();

        // Test seq scaling from 64 to 16384
        // Format: {seq_len, iterations}
        int tests[][2] = {
            {64,    100},
            {128,   100},
            {256,    50},
            {512,    50},
            {1024,   30},
            {2048,   20},
            {4096,   15},
            {8192,   10},
            {16384,   5},
        };

        int n_tests = sizeof(tests) / sizeof(tests[0]);
        double results[20];
        int seqs[20];

        printf("==========================================================\n");
        printf("%-8s %10s %10s %10s %12s\n", "seq", "time_ms", "ns/elem", "GFLOPS", "total_elem");
        printf("%-8s %10s %10s %10s %12s\n", "---", "-------", "--------", "------", "---------");

        for (int i = 0; i < n_tests; i++) {
            int seq_len = tests[i][0];
            int iter = tests[i][1];
            seqs[i] = seq_len;

            double ms = benchmark_rns_matvec(seq_len, iter);
            if (ms < 0) {
                printf("  FAILED\n");
                results[i] = -1;
                g_fail++;
            } else {
                results[i] = ms;
                g_pass++;
            }

            // Clear cache between tests to simulate fresh compilation
            orion_mil_cache_clear();
        }

        printf("\n==========================================================\n");
        printf("Summary: RNS-MatVec Scaling\n");
        printf("==========================================================\n");
        printf("%-8s %10s %10s %10s\n", "seq", "time_ms", "ns/elem", "vs_64x");
        printf("%-8s %10s %10s %10s\n", "---", "-------", "--------", "-----");

        double baseline = results[0];
        for (int i = 0; i < n_tests; i++) {
            if (results[i] > 0) {
                double ns = (results[i] * 1e6) / (N_RESIDUES * CH * seqs[i]);
                double ratio = baseline / results[i];
                printf("%-8d %10.4f %10.2f %10.2fx\n", seqs[i], results[i], ns, ratio);
            } else {
                printf("%-8d       FAILED\n", seqs[i]);
            }
        }

        printf("\n==========================================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("==========================================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}