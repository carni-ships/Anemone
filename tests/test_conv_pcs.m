// test_conv_pcs.m — Conv-PCS Test
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m core/orion_conv_pcs.m \
//     tests/test_conv_pcs.m -o test_conv_pcs
//
// Run:
//   ./test_conv_pcs

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import <CommonCrypto/CommonDigest.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "orion_mil_cache.h"
#import "orion_conv_pcs.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

#define CHECKF(cond, fmt, ...) do { \
    if (cond) { g_pass++; printf("  PASS: " fmt "\n", ##__VA_ARGS__); } \
    else { g_fail++; printf("  FAIL: " fmt "\n", ##__VA_ARGS__); } \
} while(0)

// ============================================================================
// Tests: Kernel Generation
// ============================================================================

static void test_kernel_generation(void) {
    printf("\n=== Test: Kernel Generation ===\n");

    uint8_t seed[32] = {0x01, 0x02, 0x03, 0x04};
    float kernel[CONV_PCS_KERNEL_SIZE];

    conv_pcs_generate_kernel(seed, kernel, CONV_PCS_KERNEL_SIZE);

    printf("  Kernel[0:8]: ");
    for (int i = 0; i < 8; i++) {
        printf("%.3f ", kernel[i]);
    }
    printf("\n");

    // Check values are in [-1, 1]
    bool all_bounded = true;
    for (int i = 0; i < CONV_PCS_KERNEL_SIZE; i++) {
        if (fabsf(kernel[i]) > 1.0f) all_bounded = false;
    }
    CHECK(all_bounded, "Kernel values in [-1, 1]");

    // Same seed should produce same kernel
    float kernel2[CONV_PCS_KERNEL_SIZE];
    conv_pcs_generate_kernel(seed, kernel2, CONV_PCS_KERNEL_SIZE);
    bool same = true;
    for (int i = 0; i < CONV_PCS_KERNEL_SIZE; i++) {
        if (kernel[i] != kernel2[i]) same = false;
    }
    CHECK(same, "Same seed produces same kernel");
}

// ============================================================================
// Tests: Commitment
// ============================================================================

static void test_commit_basic(void) {
    printf("\n=== Test: Commitment Basic ===\n");

    // Polynomial P(x) = 1 + 2x + 3x^2
    float poly[4] = {1.0f, 2.0f, 3.0f, 0.0f};
    int degree = 2;

    uint8_t seed[32] = {0xDE, 0xAD, 0xBE, 0xEF};
    float kernel[CONV_PCS_KERNEL_SIZE];
    conv_pcs_generate_kernel(seed, kernel, CONV_PCS_KERNEL_SIZE);

    ConvPCSCommitment commitment;
    bool ok = conv_pcs_commit(poly, degree, kernel, CONV_PCS_KERNEL_SIZE, &commitment);
    CHECK(ok, "conv_pcs_commit succeeds");

    if (ok) {
        printf("  Poly commit: ");
        for (int i = 0; i < 8; i++) printf("%02X", commitment.poly_commit[i]);
        printf("...\n");
        printf("  Merkle root: ");
        for (int i = 0; i < 8; i++) printf("%02X", commitment.merkle_root[i]);
        printf("...\n");

        // Commitment should be non-zero
        bool non_zero = false;
        for (int i = 0; i < 32; i++) {
            if (commitment.poly_commit[i] != 0) non_zero = true;
        }
        CHECK(non_zero, "Commitment is non-zero");
    }
}

static void test_commit_batched(void) {
    printf("\n=== Test: Commitment Batched ===\n");

    float polys[3 * 5] = {  // 3 polynomials, each degree 4
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // P1 = 1 + x + x^2 + x^3 + x^4
        2.0f, 0.0f, 2.0f, 0.0f, 2.0f,  // P2 = 2 + 2x^2 + 2x^4
        0.5f, 1.5f, 0.5f, 1.5f, 0.5f   // P3 = 0.5 + 1.5x + 0.5x^2 + 1.5x^3 + 0.5x^4
    };

    uint8_t seed[32] = {0xAB, 0xCD};
    float kernel[CONV_PCS_KERNEL_SIZE];
    conv_pcs_generate_kernel(seed, kernel, CONV_PCS_KERNEL_SIZE);

    ConvPCSCommitment commitments[3];
    bool ok = conv_pcs_commit_batched(polys, 3, 4, kernel, CONV_PCS_KERNEL_SIZE, commitments);
    CHECK(ok, "conv_pcs_commit_batched succeeds");

    if (ok) {
        for (int i = 0; i < 3; i++) {
            printf("  Commitment[%d]: ", i);
            for (int j = 0; j < 6; j++) printf("%02X", commitments[i].poly_commit[j]);
            printf("...\n");
        }

        // All commitments should be different
        bool all_different = true;
        for (int i = 1; i < 3; i++) {
            if (memcmp(commitments[0].poly_commit, commitments[i].poly_commit, 32) == 0) {
                all_different = false;
            }
        }
        CHECK(all_different, "Different polys produce different commits");
    }
}

// ============================================================================
// Tests: Serialization
// ============================================================================

static void test_commit_serialize(void) {
    printf("\n=== Test: Commitment Serialization ===\n");

    uint8_t seed[32] = {0x11, 0x22};
    float kernel[CONV_PCS_KERNEL_SIZE];
    conv_pcs_generate_kernel(seed, kernel, CONV_PCS_KERNEL_SIZE);

    float poly[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ConvPCSCommitment orig, restored;
    memset(&orig, 0xAA, sizeof(orig));
    memset(&restored, 0xBB, sizeof(restored));

    bool ok = conv_pcs_commit(poly, 4, kernel, CONV_PCS_KERNEL_SIZE, &orig);
    CHECK(ok, "Original commitment created");

    uint8_t serialized[64];
    conv_pcs_commit_serialize(&orig, serialized);

    conv_pcs_commit_deserialize(serialized, &restored);

    CHECK(memcmp(orig.poly_commit, restored.poly_commit, 32) == 0, "Poly commit preserved");
    CHECK(memcmp(orig.merkle_root, restored.merkle_root, 32) == 0, "Merkle root preserved");
}

// ============================================================================
// Tests: Open/Verify
// ============================================================================

static void test_open_verify(void) {
    printf("\n=== Test: Open/Verify ===\n");

    uint8_t seed[32] = {0xCA, 0xFE};
    float kernel[CONV_PCS_KERNEL_SIZE];
    conv_pcs_generate_kernel(seed, kernel, CONV_PCS_KERNEL_SIZE);

    // Polynomial P(x) = 1 + 2x + 3x^2
    float poly[4] = {1.0f, 2.0f, 3.0f, 0.0f};
    int degree = 2;

    ConvPCSCommitment commitment;
    bool ok = conv_pcs_commit(poly, degree, kernel, CONV_PCS_KERNEL_SIZE, &commitment);
    CHECK(ok, "Commitment created");

    if (ok) {
        float point = 2.0f;
        float evaluation = 1.0f + 2.0f*2.0f + 3.0f*4.0f;  // P(2) = 1 + 4 + 12 = 17

        ConvPCSProof proof;
        ok = conv_pcs_open(poly, degree, point, evaluation, kernel, CONV_PCS_KERNEL_SIZE, &proof);
        CHECK(ok, "conv_pcs_open succeeds");

        if (ok) {
            ok = conv_pcs_verify(&commitment, point, evaluation, &proof);
            CHECK(ok, "conv_pcs_verify accepts valid proof");

            // Also test the kernel-aware verification
            ok = conv_pcs_verify_with_kernel(&commitment, point, evaluation,
                                              kernel, CONV_PCS_KERNEL_SIZE, &proof);
            CHECK(ok, "conv_pcs_verify_with_kernel accepts valid proof");
        }
    }
}

// ============================================================================
// Tests: Performance
// ============================================================================

static void test_commit_performance(void) {
    printf("\n=== Test: Commitment Performance ===\n");

    uint8_t seed[32] = {0x88, 0x99};
    float kernel[CONV_PCS_KERNEL_SIZE];
    conv_pcs_generate_kernel(seed, kernel, CONV_PCS_KERNEL_SIZE);

    float poly[16];
    for (int i = 0; i < 16; i++) poly[i] = (float)(i % 10);

    const int iterations = 100;
    clock_t start = clock();

    for (int i = 0; i < iterations; i++) {
        ConvPCSCommitment c;
        conv_pcs_commit(poly, 15, kernel, CONV_PCS_KERNEL_SIZE, &c);
    }

    clock_t end = clock();
    double total_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / iterations;

    printf("  %d iterations: %.2f ms total, %.4f ms/iter\n",
           iterations, total_ms, per_iter_ms);

    CHECKF(per_iter_ms < 10.0, "Commit under 10ms per iteration (CPU fallback)");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Conv-PCS Test ===\n");
        printf("Kernel size: %d, Max degree: %d\n\n",
               CONV_PCS_KERNEL_SIZE, CONV_PCS_MAX_DEGREE);

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        orion_mil_cache_clear();

        test_kernel_generation();
        test_commit_basic();
        test_commit_batched();
        test_commit_serialize();
        test_open_verify();
        test_commit_performance();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        printf("\n=== Conv-PCS Summary ===\n");
        printf("Convolution-based polynomial commitment\n");
        printf("Commitment: hash of polynomial evaluation\n");
        printf("Opening: proof of correct evaluation\n");
        printf("Next: Full Merkle tree integration\n");

        return g_fail > 0 ? 1 : 0;
    }
}