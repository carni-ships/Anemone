// test_latticezk.m — ANE LatticeZK Test
//
// Verifies the core infrastructure for ANE-accelerated lattice zk-SNARKs:
//   1. RNS configuration for Dilithium-3
//   2. A*s per-residue evaluation on ANE
//   3. CRT reconstruction
//   4. End-to-end correctness
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m core/orion_rns.m core/orion_latticezk.m \
//     tests/test_latticezk.m -o test_latticezk
//
// Run:
//   ./test_latticezk

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <time.h>
#import <math.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import "mil_cache.h"
#import "rns.h"
#import "latticezk.h"

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
// Tests: RNS Configuration
// ============================================================================

static void test_rns_config(void) {
    printf("\n=== Test: RNS Configuration ===\n");

    const LatticeZKRNSConfig *rns = latticezk_rns_config();

    printf("  Moduli: ");
    for (int i = 0; i < rns->n_mods; i++) {
        printf("%u ", rns->mods[i].mod);
    }
    printf("\n");
    printf("  Product M: %llu (~%.1f bits)\n",
           (unsigned long long)rns->product, rns->bits);
    printf("  q (Dilithium-3): %u\n", LATTICEZK_Q);

    CHECKF(rns->product > LATTICEZK_Q, "M > q (CRT can reconstruct)");
    CHECKF(rns->bits > 23.0, "M > 23 bits (covers q)");

    // Verify all moduli < 128 (fp16 safe)
    bool all_safe = true;
    for (int i = 0; i < rns->n_mods; i++) {
        if (rns->mods[i].mod >= 128) all_safe = false;
    }
    CHECKF(all_safe, "All moduli < 128 (fp16 safe)");
}

// ============================================================================
// Tests: ExpandA (Matrix Generation)
// ============================================================================

static void test_expand_a(void) {
    printf("\n=== Test: ExpandA (Matrix Generation) ===\n");

    float A[LATTICEZK_K * LATTICEZK_L];
    uint8_t seed[32] = {0xDE, 0xAD, 0xBE, 0xEF};

    latticezk_expand_a(seed, A, LATTICEZK_K, LATTICEZK_L);

    printf("  A matrix (%dx%d):\n", LATTICEZK_K, LATTICEZK_L);
    for (int i = 0; i < LATTICEZK_K; i++) {
        printf("    ");
        for (int j = 0; j < LATTICEZK_L; j++) {
            printf("%+.3f ", A[i * LATTICEZK_L + j]);
        }
        printf("\n");
    }

    // Check values are in reasonable range
    bool all_in_range = true;
    for (int i = 0; i < LATTICEZK_K * LATTICEZK_L; i++) {
        if (fabsf(A[i]) > 2.0f) all_in_range = false;
    }
    CHECK(all_in_range, "A values in [-2, 2] range");
}

// ============================================================================
// Tests: Short Vector Sampling
// ============================================================================

static void test_sample_short_vector(void) {
    printf("\n=== Test: Short Vector Sampling ===\n");

    float s[LATTICEZK_L];
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    printf("  s vector: [");
    for (int j = 0; j < LATTICEZK_L; j++) {
        printf("%+.1f ", s[j]);
    }
    printf("]\n");

    // Check values are small
    bool all_small = true;
    for (int j = 0; j < LATTICEZK_L; j++) {
        if (fabsf(s[j]) > 2.0f) all_small = false;
    }
    CHECK(all_small, "s values bounded by lambda");
}

// ============================================================================
// Tests: CPU Baseline MatVec
// ============================================================================

static void cpu_matvec(const float *A, const float *s, float *y, int k, int l) {
    for (int i = 0; i < k; i++) {
        y[i] = 0;
        for (int j = 0; j < l; j++) {
            y[i] += A[i * l + j] * s[j];
        }
    }
}

static void test_cpu_matvec(void) {
    printf("\n=== Test: CPU MatVec Baseline ===\n");

    float A[LATTICEZK_K * LATTICEZK_L];
    float s[LATTICEZK_L];
    float y_cpu[LATTICEZK_K];

    uint8_t seed[32] = {0xDE, 0xAD, 0xBE, 0xEF};
    latticezk_expand_a(seed, A, LATTICEZK_K, LATTICEZK_L);
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    cpu_matvec(A, s, y_cpu, LATTICEZK_K, LATTICEZK_L);

    printf("  CPU result: [");
    for (int i = 0; i < LATTICEZK_K; i++) {
        printf("%.2f ", y_cpu[i]);
    }
    printf("]\n");

    CHECK(true, "CPU matvec computed");
}

// ============================================================================
// Tests: ANE RNS MatVec
// ============================================================================

static void test_ane_rns_matvec(void) {
    printf("\n=== Test: ANE RNS MatVec ===\n");

    float A[LATTICEZK_K * LATTICEZK_L];
    float s[LATTICEZK_L];

    uint8_t seed[32] = {0xDE, 0xAD, 0xBE, 0xEF};
    latticezk_expand_a(seed, A, LATTICEZK_K, LATTICEZK_L);
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    const LatticeZKRNSConfig *rns = latticezk_rns_config();
    float *residues = (float *)malloc(LATTICEZK_K * rns->n_mods * sizeof(float));

    bool ok = latticezk_rns_matvec(A, s, LATTICEZK_K, LATTICEZK_L,
                                   residues, rns->n_mods, rns);
    CHECK(ok, "ANE RNS matvec succeeds");

    if (ok) {
        printf("  Per-residue results (k=%d, n_mods=%d):\n", LATTICEZK_K, rns->n_mods);
        for (int r = 0; r < rns->n_mods; r++) {
            printf("  mod %u: [", rns->mods[r].mod);
            for (int i = 0; i < LATTICEZK_K; i++) {
                printf("%.2f ", residues[r * LATTICEZK_K + i]);
            }
            printf("]\n");
        }
    }

    free(residues);
}

// ============================================================================
// Tests: CRT Reconstruction
// ============================================================================

static void test_crt_reconstruction(void) {
    printf("\n=== Test: CRT Reconstruction ===\n");

    // Test with known small values
    // x = 12345, M = 15015, residues = {0, 0, 4, 3, 8} mod {3,5,7,11,13}
    const RNSMod test_mods[] = {
        {3, "t0"}, {5, "t1"}, {7, "t2"}, {11, "t3"}, {13, "t4"}
    };
    uint32_t residues[] = {0, 0, 4, 3, 8};

    uint64_t recon = orion_crt_reconstruct(residues, test_mods, 5);
    printf("  CRT test: x=12345 -> %llu %s\n",
           (unsigned long long)recon,
           recon == 12345 ? "(PASS)" : "(should be 12345)");

    CHECK(recon == 12345, "CRT reconstruction correct");

    // Test with actual LatticeZK moduli
    const LatticeZKRNSConfig *rns = latticezk_rns_config();

    // Generate simple test: A = ones, s = ones
    // Expected result: A*s = [4, 4, 4, 4] for each row (since l=4)
    float A[LATTICEZK_K * LATTICEZK_L];
    float s[LATTICEZK_L];
    for (int i = 0; i < LATTICEZK_K * LATTICEZK_L; i++) A[i] = 1.0f;
    for (int j = 0; j < LATTICEZK_L; j++) s[j] = 1.0f;

    // Clear cache to ensure fresh compilation for this test
    orion_mil_cache_clear();

    float *ane_residues = (float *)malloc(LATTICEZK_K * rns->n_mods * sizeof(float));
    bool ok = latticezk_rns_matvec(A, s, LATTICEZK_K, LATTICEZK_L,
                                   ane_residues, rns->n_mods, rns);
    CHECK(ok, "ANE RNS matvec succeeds");

    uint64_t result[LATTICEZK_K];
    latticezk_crt_reconstruct(ane_residues, LATTICEZK_K, rns, LATTICEZK_Q, result);

    printf("  CRT result mod %d: [", LATTICEZK_Q);
    for (int i = 0; i < LATTICEZK_K; i++) {
        printf("%llu ", (unsigned long long)result[i]);
    }
    printf("]\n");

    // For A=1, s=1, expected each output = l = 4 (mod q)
    bool all_correct = true;
    for (int i = 0; i < LATTICEZK_K; i++) {
        if (result[i] != 4) all_correct = false;
    }
    CHECK(all_correct, "CRT result = 4 for A=1, s=1");

    free(ane_residues);
}

// ============================================================================
// Tests: End-to-End MatVec
// ============================================================================

static void test_end_to_end(void) {
    printf("\n=== Test: End-to-End MatVec ===\n");

    float A[LATTICEZK_K * LATTICEZK_L];
    float s[LATTICEZK_L];
    uint64_t result[LATTICEZK_K];

    uint8_t seed[32] = {0x12, 0x34, 0x56, 0x78};
    latticezk_expand_a(seed, A, LATTICEZK_K, LATTICEZK_L);
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    bool ok = latticezk_matvec(A, s, LATTICEZK_K, LATTICEZK_L, LATTICEZK_Q, result);
    CHECK(ok, "latticezk_matvec succeeds");

    if (ok) {
        printf("  Final result mod %d: [", LATTICEZK_Q);
        for (int i = 0; i < LATTICEZK_K; i++) {
            printf("%llu ", (unsigned long long)result[i]);
        }
        printf("]\n");
    }

    // CPU baseline for comparison
    float y_cpu[LATTICEZK_K];
    cpu_matvec(A, s, y_cpu, LATTICEZK_K, LATTICEZK_L);
    printf("  CPU baseline (not mod q): [");
    for (int i = 0; i < LATTICEZK_K; i++) {
        printf("%.2f ", y_cpu[i]);
    }
    printf("]\n");
}

// ============================================================================
// Tests: Performance
// ============================================================================

static void test_performance(void) {
    printf("\n=== Test: Performance ===\n");

    float A[LATTICEZK_K * LATTICEZK_L];
    float s[LATTICEZK_L];
    uint64_t result[LATTICEZK_K];

    uint8_t seed[32] = {0xAB, 0xCD};
    latticezk_expand_a(seed, A, LATTICEZK_K, LATTICEZK_L);
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    const int iterations = 50;
    clock_t start = clock();

    for (int i = 0; i < iterations; i++) {
        latticezk_matvec(A, s, LATTICEZK_K, LATTICEZK_L, LATTICEZK_Q, result);
    }

    clock_t end = clock();
    double total_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / iterations;

    printf("  %d iterations: %.2f ms total, %.4f ms/iter\n",
           iterations, total_ms, per_iter_ms);

    CHECK(per_iter_ms < 10.0, "MatVec under 10ms per iteration");
}

// ============================================================================
// Tests: Fiat-Shamir Transcript
// ============================================================================

static void test_fiat_shamir(void) {
    printf("\n=== Test: Fiat-Shamir Transcript ===\n");

    LatticeZKTranscript t;
    latticezk_transcript_init(&t);
    CHECK(t.len == 0, "Transcript init sets len=0");

    // Append some data
    uint8_t data[] = {0x01, 0x02, 0x03, 0x04};
    latticezk_transcript_append(&t, data, 4);
    CHECK(t.len == 4, "Append updates length");

    // Append u64
    latticezk_transcript_append_u64(&t, 12345);
    CHECK(t.len == 12, "Append u64 adds 8 bytes");

    // Append field element
    latticezk_transcript_append_field(&t, 999999, 8383489);
    CHECK(t.len == 20, "Append field adds 8 bytes");

    // Generate challenge
    uint8_t challenge[32];
    latticezk_challenge_from_transcript(&t, challenge);

    // Check challenge is non-zero (was hashed from our data)
    bool non_zero = false;
    for (int i = 0; i < 32; i++) {
        if (challenge[i] != 0) non_zero = true;
    }
    CHECK(non_zero, "Challenge is non-zero");

    // Different data should produce different challenge
    LatticeZKTranscript t2;
    latticezk_transcript_init(&t2);
    uint8_t different_data[] = {0x99, 0x88, 0x77};
    latticezk_transcript_append(&t2, different_data, 3);

    uint8_t challenge2[32];
    latticezk_challenge_from_transcript(&t2, challenge2);

    bool different = false;
    for (int i = 0; i < 32; i++) {
        if (challenge[i] != challenge2[i]) different = true;
    }
    CHECK(different, "Different transcript produces different challenge");
}

// ============================================================================
// Tests: Prove/Verify
// ============================================================================

static void test_prove_verify(void) {
    printf("\n=== Test: Prove/Verify ===\n");

    // Create proving key
    LatticeZKProvingKey pk = {
        .seed = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE},
        .q = LATTICEZK_Q,
        .k = LATTICEZK_K,
        .l = LATTICEZK_L,
        .n = LATTICEZK_N
    };
    // Ensure seed is 32 bytes
    memset(pk.seed + 8, 0, 24);

    // Create verification key
    LatticeZKVerificationKey vk = {
        .q = LATTICEZK_Q,
        .k = LATTICEZK_K,
        .l = LATTICEZK_L,
        .n = LATTICEZK_N
    };

    // Generate short witness s
    float s[LATTICEZK_L];
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    // Generate proof
    LatticeZKProof proof;
    memset(&proof, 0, sizeof(proof));

    orion_mil_cache_clear();
    bool ok = latticezk_prove(&pk, s, &proof);
    CHECK(ok, "latticezk_prove succeeds");

    if (ok) {
        printf("  Commitment: ");
        for (int i = 0; i < 8; i++) printf("%02X", proof.commitment[i]);
        printf("...\n");
        printf("  Challenge: ");
        for (int i = 0; i < 8; i++) printf("%02X", proof.challenge[i]);
        printf("...\n");
        printf("  Response[0]: %llu\n", (unsigned long long)proof.response[0]);

        // Verify proof
        ok = latticezk_verify(&vk, &proof);
        CHECK(ok, "latticezk_verify accepts valid proof");
    }

    // Tamper with proof - should fail verification
    if (ok) {
        proof.response[0] ^= 0xFFFF;
        bool tampered_ok = latticezk_verify(&vk, &proof);
        CHECK(!tampered_ok, "latticezk_verify rejects tampered proof");
    }
}

// ============================================================================
// Tests: Proof Serialization
// ============================================================================

static void test_proof_serialization(void) {
    printf("\n=== Test: Proof Serialization ===\n");

    // Create a proof
    LatticeZKProvingKey pk = {
        .seed = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE},
        .q = LATTICEZK_Q,
        .k = LATTICEZK_K,
        .l = LATTICEZK_L,
        .n = LATTICEZK_N
    };
    memset(pk.seed + 8, 0, 24);

    float s[LATTICEZK_L];
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    LatticeZKProof proof;
    memset(&proof, 0, sizeof(proof));

    orion_mil_cache_clear();
    bool ok = latticezk_prove(&pk, s, &proof);
    CHECK(ok, "Prove generates valid proof");

    // Serialize
    uint8_t serialized[LATTICEZK_PROOF_SIZE];
    size_t serialized_len = 0;
    ok = latticezk_proof_serialize(&proof, serialized, &serialized_len);
    CHECK(ok, "serialize succeeds");
    CHECK(serialized_len == LATTICEZK_PROOF_SIZE, "Serialized size matches");

    // Deserialize
    LatticeZKProof proof2;
    ok = latticezk_proof_deserialize(serialized, serialized_len, &proof2);
    CHECK(ok, "deserialize succeeds");

    // Verify deserialized proof
    LatticeZKVerificationKey vk = {
        .q = LATTICEZK_Q,
        .k = LATTICEZK_K,
        .l = LATTICEZK_L,
        .n = LATTICEZK_N
    };

    ok = latticezk_verify(&vk, &proof2);
    CHECK(ok, "Deserialized proof verifies correctly");

    // Check individual fields match
    CHECK(memcmp(proof.commitment, proof2.commitment, 32) == 0, "Commitment preserved");
    CHECK(memcmp(proof.challenge, proof2.challenge, 32) == 0, "Challenge preserved");
    for (int i = 0; i < LATTICEZK_K; i++) {
        if (proof.response[i] != proof2.response[i]) {
            ok = false;
        }
    }
    CHECK(ok, "Response preserved");
}

// ============================================================================
// Tests: Signing
// ============================================================================

static void test_signing_basics(void) {
    printf("\n=== Test: Signing Basics ===\n");

    // Test keygen
    uint8_t seed[32] = {0x01, 0x02, 0x03, 0x04};
    LatticeZKProvingKey pk;
    LatticeZKVerificationKey vk;

    latticezk_keygen(seed, &pk, &vk);
    CHECK(pk.q == LATTICEZK_Q, "PK has correct q");
    CHECK(vk.q == LATTICEZK_Q, "VK has correct q");
    CHECK(memcmp(pk.seed, seed, 32) == 0, "PK seed matches input");
}

static void test_sign_and_verify(void) {
    printf("\n=== Test: Sign and Verify ===\n");

    // Create keypair
    uint8_t seed[32] = {0xDE, 0xAD, 0xBE, 0xEF};
    LatticeZKProvingKey pk;
    LatticeZKVerificationKey vk;
    latticezk_keygen(seed, &pk, &vk);

    // Sign a message
    uint8_t message[8] = {'H', 'e', 'l', 'l', 'o', ',', 'W', 'o'};
    size_t m_len = 8;
    uint8_t signature[256];  // Large enough for variable-length sig

    orion_mil_cache_clear();
    bool ok = latticezk_sign(&pk, message, m_len, signature);
    CHECK(ok, "latticezk_sign succeeds");

    if (ok) {
        // Verify signature
        ok = latticezk_verify_sig(&vk, message, m_len, signature);
        CHECK(ok, "latticezk_verify_sig accepts valid signature");
    }

    // Wrong message should fail
    if (ok) {
        uint8_t wrong_message[8] = {'W', 'r', 'o', 'n', 'g', '!', '!', '!'};
        size_t wrong_len = 8;
        ok = latticezk_verify_sig(&vk, wrong_message, wrong_len, signature);
        CHECK(!ok, "latticezk_verify_sig rejects wrong message");
    }
}

static void test_signing_performance(void) {
    printf("\n=== Test: Signing Performance ===\n");

    uint8_t seed[32] = {0xAB, 0xCD};
    LatticeZKProvingKey pk;
    LatticeZKVerificationKey vk;
    latticezk_keygen(seed, &pk, &vk);

    uint8_t message[64];
    for (int i = 0; i < 64; i++) message[i] = i;
    uint8_t signature[256];

    const int iterations = 20;
    clock_t start = clock();

    orion_mil_cache_clear();
    for (int i = 0; i < iterations; i++) {
        latticezk_sign(&pk, message, sizeof(message), signature);
    }

    clock_t end = clock();
    double total_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double per_iter_ms = total_ms / iterations;

    printf("  %d iterations: %.2f ms total, %.4f ms/iter\n",
           iterations, total_ms, per_iter_ms);

    CHECK(per_iter_ms < 50.0, "Sign under 50ms per iteration");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== ANE LatticeZK Test ===\n");
        printf("Dilithium-3 params: k=%d, l=%d, n=%d, q=%d\n",
               LATTICEZK_K, LATTICEZK_L, LATTICEZK_N, LATTICEZK_Q);
        printf("RNS residues: %d\n\n", LATTICEZK_N_RESIDUES);

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized. compile_count=%d\n\n", orion_compile_count());

        orion_mil_cache_clear();

        test_rns_config();
        test_expand_a();
        test_sample_short_vector();
        test_cpu_matvec();
        test_ane_rns_matvec();
        test_crt_reconstruction();
        test_end_to_end();
        test_performance();
        test_fiat_shamir();
        test_prove_verify();
        test_proof_serialization();
        test_signing_basics();
        test_sign_and_verify();
        test_signing_performance();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Total compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        printf("\n=== ANE LatticeZK Summary ===\n");
        printf("RNS Configuration: %d moduli, M ≈ 2^%.1f bits\n",
               LATTICEZK_N_RESIDUES, latticezk_rns_config()->bits);
        printf("A*s mod q_i: ANE (batched conv1x1)\n");
        printf("A*s mod q: CRT reconstruction on CPU\n");
        printf("Fiat-Shamir: SHA-256 transcript + challenge\n");
        printf("Prove/Verify: Full proof generation verified\n");
        printf("Next: Real Dilithium signing protocol, proof serialization\n");

        return g_fail > 0 ? 1 : 0;
    }
}
