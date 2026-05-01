// test_gpu_ntt.m — GPU NTT Test
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework Metal -ldl \
//     -I . -I core \
//     core/orion_gpu_ntt.m tests/test_gpu_ntt.m -o test_gpu_ntt
//
// Run:
//   ./test_gpu_ntt

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import "orion_gpu_ntt.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

#define CHECKF(cond, fmt, ...) do { \
    if (cond) { g_pass++; printf("  PASS: " fmt "\n", ##__VA_ARGS__); } \
    else { g_fail++; printf("  FAIL: " fmt "\n", ##__VA_ARGS__); } \
} while(0)

static uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)((uint64_t)a * b % q);
}

static uint32_t mod_pow_cpu(uint32_t a, uint64_t e, uint32_t q) {
    uint64_t r = 1;
    uint64_t base = a % q;
    while (e > 0) {
        if (e & 1) r = (r * base) % q;
        base = (base * base) % q;
        e >>= 1;
    }
    return (uint32_t)r;
}

static void cpu_ntt_forward(GPUNTTPoly *poly, uint32_t q, uint32_t w) {
    int n = GPU_NTT_N;
    for (int len = 2; len <= n; len <<= 1) {
        uint32_t w_local = mod_pow_cpu(w, (q - 1) / len, q);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            uint32_t w_power = 1;
            for (int j = 0; j < half; j++) {
                uint32_t u = poly->coeff[i + j];
                uint32_t v = mod_mul(poly->coeff[i + j + half], w_power, q);
                poly->coeff[i + j] = u + v;
                if (poly->coeff[i + j] >= q) poly->coeff[i + j] -= q;
                poly->coeff[i + j + half] = u + q - v;
                if (poly->coeff[i + j + half] >= q) poly->coeff[i + j + half] -= q;
                w_power = mod_mul(w_power, w_local, q);
            }
        }
    }
}

static void cpu_ntt_inverse(GPUNTTPoly *poly, uint32_t q, uint32_t w) {
    int n = GPU_NTT_N;
    // Inverse uses w^(-1) and scaled by n^(-1)
    uint32_t w_inv = mod_pow_cpu(w, q - 2, q);
    uint32_t n_inv = mod_pow_cpu(n, q - 2, q);

    for (int len = 2; len <= n; len <<= 1) {
        uint32_t w_local = mod_pow_cpu(w_inv, (q - 1) / len, q);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            uint32_t w_power = 1;
            for (int j = 0; j < half; j++) {
                uint32_t u = poly->coeff[i + j];
                uint32_t v = mod_mul(poly->coeff[i + j + half], w_power, q);
                poly->coeff[i + j] = u + v;
                if (poly->coeff[i + j] >= q) poly->coeff[i + j] -= q;
                poly->coeff[i + j + half] = u + q - v;
                if (poly->coeff[i + j + half] >= q) poly->coeff[i + j + half] -= q;
                w_power = mod_mul(w_power, w_local, q);
            }
        }
    }

    // Scale by n^(-1)
    for (int i = 0; i < n; i++) {
        poly->coeff[i] = mod_mul(poly->coeff[i], n_inv, q);
    }
}

static void cpu_poly_mult(GPUNTTPoly *result, const GPUNTTPoly *a, const GPUNTTPoly *b, uint32_t q, uint32_t w) {
    GPUNTTPoly fa, fb;
    memcpy(&fa, a, sizeof(GPUNTTPoly));
    memcpy(&fb, b, sizeof(GPUNTTPoly));

    cpu_ntt_forward(&fa, q, w);
    cpu_ntt_forward(&fb, q, w);

    for (int i = 0; i < GPU_NTT_N; i++) {
        fa.coeff[i] = mod_mul(fa.coeff[i], fb.coeff[i], q);
    }

    cpu_ntt_inverse(&fa, q, w);
    memcpy(result, &fa, sizeof(GPUNTTPoly));
}

static void test_gpu_available(void) {
    printf("\n=== Test: GPU Available ===\n");
    bool available = orion_gpu_available();
    CHECK(available, "GPU is available");

    if (available) {
        O_RIONGPUContext *ctx = orion_gpu_init();
        if (ctx) {
            printf("  Device: %s\n", orion_gpu_device_name(ctx));
            orion_gpu_release(ctx);
        }
    }
}

static void test_ntt_forward(void) {
    printf("\n=== Test: NTT Forward ===\n");

    O_RIONGPUContext *ctx = orion_gpu_init();
    if (!ctx) {
        printf("  SKIP: No GPU context\n");
        return;
    }

    GPUNTTPoly input, gpu_output, cpu_output;
    memset(&input, 0, sizeof(input));
    memset(&gpu_output, 0, sizeof(gpu_output));
    memset(&cpu_output, 0, sizeof(cpu_output));

    // Set polynomial: p(x) = 1 + 2x + 3x^2 + ...
    for (int i = 0; i < GPU_NTT_N; i++) {
        input.coeff[i] = (i + 1) % GPU_NTT_Q;
    }

    memcpy(&cpu_output, &input, sizeof(GPUNTTPoly));
    cpu_ntt_forward(&cpu_output, GPU_NTT_Q, GPU_NTT_W);

    bool ok = orion_ntt_forward(ctx, &input, &gpu_output);
    CHECK(ok, "GPU NTT forward succeeds");

    if (ok) {
        // Compare with CPU reference
        bool match = true;
        for (int i = 0; i < GPU_NTT_N; i++) {
            if (gpu_output.coeff[i] != cpu_output.coeff[i]) {
                match = false;
                printf("  Mismatch at index %d: GPU=%u CPU=%u\n",
                       i, gpu_output.coeff[i], cpu_output.coeff[i]);
                break;
            }
        }
        CHECK(match, "GPU output matches CPU reference");
    }

    orion_gpu_release(ctx);
}

static void test_ntt_inverse(void) {
    printf("\n=== Test: NTT Inverse ===\n");

    O_RIONGPUContext *ctx = orion_gpu_init();
    if (!ctx) {
        printf("  SKIP: No GPU context\n");
        return;
    }

    GPUNTTPoly input, roundtrip, cpu_roundtrip;
    memset(&input, 0, sizeof(input));
    memset(&roundtrip, 0, sizeof(roundtrip));

    // Simple test polynomial
    input.coeff[0] = 123;
    input.coeff[1] = 456;
    input.coeff[2] = 789;
    for (int i = 3; i < GPU_NTT_N; i++) input.coeff[i] = 0;

    memcpy(&cpu_roundtrip, &input, sizeof(GPUNTTPoly));
    cpu_ntt_forward(&cpu_roundtrip, GPU_NTT_Q, GPU_NTT_W);
    cpu_ntt_inverse(&cpu_roundtrip, GPU_NTT_Q, GPU_NTT_W);

    bool ok1 = orion_ntt_forward(ctx, &input, &roundtrip);
    CHECK(ok1, "Forward pass succeeds");

    if (ok1) {
        memset(&input, 0, sizeof(input));
        memcpy(&input, &roundtrip, sizeof(GPUNTTPoly));
        memset(&roundtrip, 0, sizeof(roundtrip));
        bool ok2 = orion_ntt_inverse(ctx, &input, &roundtrip);
        CHECK(ok2, "Inverse pass succeeds");

        if (ok2) {
            bool match = true;
            for (int i = 0; i < GPU_NTT_N; i++) {
                if (roundtrip.coeff[i] != cpu_roundtrip.coeff[i]) {
                    match = false;
                    break;
                }
            }
            CHECK(match, "Roundtrip matches CPU reference");
        }
    }

    orion_gpu_release(ctx);
}

static void test_ntt_multiply(void) {
    printf("\n=== Test: NTT Multiply ===\n");

    O_RIONGPUContext *ctx = orion_gpu_init();
    if (!ctx) {
        printf("  SKIP: No GPU context\n");
        return;
    }

    GPUNTTPoly a, b, gpu_result, cpu_result;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&gpu_result, 0, sizeof(gpu_result));
    memset(&cpu_result, 0, sizeof(cpu_result));

    // a(x) = 1 + x
    a.coeff[0] = 1;
    a.coeff[1] = 1;
    for (int i = 2; i < GPU_NTT_N; i++) a.coeff[i] = 0;

    // b(x) = 2 + 3x
    b.coeff[0] = 2;
    b.coeff[1] = 3;
    for (int i = 2; i < GPU_NTT_N; i++) b.coeff[i] = 0;

    bool ok = orion_ntt_multiply(ctx, &a, &b, &gpu_result);
    CHECK(ok, "GPU multiply succeeds");

    if (ok) {
        cpu_poly_mult(&cpu_result, &a, &b, GPU_NTT_Q, GPU_NTT_W);

        bool match = true;
        for (int i = 0; i < GPU_NTT_N; i++) {
            if (gpu_result.coeff[i] != cpu_result.coeff[i]) {
                match = false;
                printf("  Mismatch at index %d: GPU=%u CPU=%u\n",
                       i, gpu_result.coeff[i], cpu_result.coeff[i]);
            }
        }
        CHECK(match, "GPU multiply matches CPU reference");
    }

    orion_gpu_release(ctx);
}

static void test_batch_ntt(void) {
    printf("\n=== Test: Batch NTT ===\n");

    O_RIONGPUContext *ctx = orion_gpu_init();
    if (!ctx) {
        printf("  SKIP: No GPU context\n");
        return;
    }

    const int count = 4;
    GPUNTTPoly inputs[count], outputs[count];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    for (int i = 0; i < count; i++) {
        inputs[i].coeff[0] = i + 1;
        inputs[i].coeff[1] = (i + 1) * 2;
    }

    bool ok = orion_ntt_forward_batch(ctx, inputs, outputs, count);
    CHECK(ok, "Batch forward succeeds");

    if (ok) {
        bool all_nonzero = true;
        for (int i = 0; i < count; i++) {
            bool has_nonzero = false;
            for (int j = 0; j < GPU_NTT_N; j++) {
                if (outputs[i].coeff[j] != 0) {
                    has_nonzero = true;
                    break;
                }
            }
            if (!has_nonzero) all_nonzero = false;
        }
        CHECK(all_nonzero, "All batch outputs are non-zero (transformed)");
    }

    orion_gpu_release(ctx);
}

static void test_ring_arithmetic(void) {
    printf("\n=== Test: Ring Arithmetic ===\n");

    GPUNTTPoly a, b, c;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&c, 0, sizeof(c));

    a.coeff[0] = 100;
    b.coeff[0] = 200;

    orion_poly_add(&c, &a, &b);
    CHECKF(c.coeff[0] == 300, "Add: 100 + 200 = 300");

    a.coeff[0] = 100;
    b.coeff[0] = 200;
    orion_poly_sub(&c, &a, &b);
    CHECKF(c.coeff[0] == (100 + GPU_NTT_Q - 200) % GPU_NTT_Q, "Sub: 100 - 200 mod q");

    a.coeff[0] = 10;
    orion_poly_scalar_mul(&c, 5, &a);
    CHECKF(c.coeff[0] == 50, "Scalar: 10 * 5 = 50");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== GPU NTT Test ===\n");
        printf("Ring: q=%d, n=%d, w=%d\n\n", GPU_NTT_Q, GPU_NTT_N, GPU_NTT_W);

        test_gpu_available();
        test_ring_arithmetic();
        test_ntt_forward();
        test_ntt_inverse();
        test_ntt_multiply();
        test_batch_ntt();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}