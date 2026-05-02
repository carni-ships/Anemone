// test_verification.c — Empirical verification for Labrador protocol
// Run on ANE hardware to verify completeness, soundness, and ANE/CPU consistency

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "core/latticezk.h"

// Number of test iterations for property-based testing
#define N_TESTS 10000
#define N_TAMPERED_TESTS 100

// Test result tracking
typedef struct {
    int completeness_pass;
    int completeness_fail;
    int soundness_pass;
    int soundness_fail;
    int cpu_consistency_pass;
    int cpu_consistency_fail;
} TestResults;

void init_results(TestResults *r) {
    r->completeness_pass = 0;
    r->completeness_fail = 0;
    r->soundness_pass = 0;
    r->soundness_fail = 0;
    r->cpu_consistency_pass = 0;
    r->cpu_consistency_fail = 0;
}

// Generate random bytes for seed
void random_seed(uint8_t *seed, int len) {
    for (int i = 0; i < len; i++) {
        seed[i] = (uint8_t)(rand() & 0xFF);
    }
}

// Generate random short vector for witness
void random_short_vector(float *s, int l, float lambda) {
    for (int i = 0; i < l; i++) {
        int sign = (rand() % 2) ? 1 : -1;
        float mag = (float)(rand() % (int)(lambda * 100)) / 100.0f;
        if (mag < 0.1f) mag = 0.1f;  // Ensure non-zero
        s[i] = sign * mag;
    }
}

// Simplified CPU MatVec (for consistency testing)
void cpu_matvec(const float *A, const float *s, int k, int l, uint64_t q, uint64_t *result) {
    for (int i = 0; i < k; i++) {
        uint64_t sum = 0;
        for (int j = 0; j < l; j++) {
            sum += (uint64_t)(A[i * l + j] * s[j]);
        }
        result[i] = sum % q;
    }
}

// Test 1: Completeness - honest prover proofs should always verify
int test_completeness(TestResults *r) {
    printf("=== Completeness Test (%d iterations) ===\n", N_TESTS);

    for (int t = 0; t < N_TESTS; t++) {
        LatticeZKProvingKey pk;
        LatticeZKVerificationKey vk;
        LatticeZKProof proof;

        // Random seed
        uint8_t seed[32];
        random_seed(seed, 32);

        // Keygen
        latticezk_keygen(seed, &pk, &vk);

        // Random short witness
        float s[LATTICEZK_L];
        random_short_vector(s, LATTICEZK_L, 2.0f);

        // Prove
        if (!latticezk_prove(&pk, s, &proof)) {
            r->completeness_fail++;
            printf("  FAIL: prove() returned false at iteration %d\n", t);
            continue;
        }

        // Verify
        if (latticezk_verify(&vk, &proof)) {
            r->completeness_pass++;
        } else {
            r->completeness_fail++;
            printf("  FAIL: honest proof %d rejected (this is bad!)\n", t);
        }
    }

    printf("  Result: %d/%d passed (%.2f%%)\n\n",
           r->completeness_pass, r->completeness_pass + r->completeness_fail,
           100.0 * r->completeness_pass / (r->completeness_pass + r->completeness_fail));

    return r->completeness_fail == 0 ? 0 : 1;
}

// Test 2: Soundness - tampered proofs should be rejected
int test_soundness(TestResults *r) {
    printf("=== Soundness Test (%d iterations) ===\n", N_TESTS);

    for (int t = 0; t < N_TESTS; t++) {
        LatticeZKProvingKey pk;
        LatticeZKVerificationKey vk;
        LatticeZKProof proof;

        uint8_t seed[32];
        random_seed(seed, 32);
        latticezk_keygen(seed, &pk, &vk);

        float s[LATTICEZK_L];
        random_short_vector(s, LATTICEZK_L, 2.0f);

        if (!latticezk_prove(&pk, s, &proof)) {
            continue;  // Skip failed proofs
        }

        // Choose random tamper method
        int tamper = rand() % 4;
        int tamper_idx = rand() % LATTICEZK_K;

        switch (tamper) {
            case 0:  // Tamper commitment
                proof.commitment[rand() % 32] ^= 0xFF;
                break;
            case 1:  // Tamper challenge
                proof.challenge[rand() % 32] ^= 0xFF;
                break;
            case 2:  // Tamper response out of bounds
                proof.response[tamper_idx] = pk.q + (rand() % 1000);  // Invalid: >= q
                break;
            case 3:  // Tamper response to different value
                proof.response[tamper_idx] = (proof.response[tamper_idx] + 1) % pk.q;
                break;
        }

        if (!latticezk_verify(&vk, &proof)) {
            r->soundness_pass++;  // Correctly rejected
        } else {
            r->soundness_fail++;
            printf("  FAIL: tampered proof %d (method %d) accepted (forgery possible!)\n", t, tamper);
        }
    }

    printf("  Result: %d/%d correctly rejected (%.2f%%)\n\n",
           r->soundness_pass, r->soundness_pass + r->soundness_fail,
           100.0 * r->soundness_pass / (r->soundness_pass + r->soundness_fail));

    // Soundness requires >99% rejection of tampered proofs
    float rejection_rate = (float)r->soundness_pass / (r->soundness_pass + r->soundness_fail);
    return rejection_rate > 0.99 ? 0 : 1;
}

// Test 3: ANE/CPU Consistency (requires ANE hardware)
int test_ane_cpu_consistency(TestResults *r) {
    printf("=== ANE/CPU Consistency Test (%d iterations) ===\n", N_TESTS);

    // Note: This test requires actual ANE hardware
    // Without ANE, latticezk_matvec will fall back to CPU path

    for (int t = 0; t < N_TESTS; t++) {
        float A[LATTICEZK_K * LATTICEZK_L];
        float s[LATTICEZK_L];
        uint64_t result_ane[LATTICEZK_K];
        uint64_t result_cpu[LATTICEZK_K];

        // Generate random matrix and vector
        for (int i = 0; i < LATTICEZK_K * LATTICEZK_L; i++) {
            A[i] = ((float)(rand() % 100) - 50.0f) / 50.0f;  // [-1, 1] range
        }
        for (int i = 0; i < LATTICEZK_L; i++) {
            s[i] = ((float)(rand() % 100) - 50.0f) / 50.0f;  // [-1, 1] range
        }

        // Compute via ANE (or CPU fallback)
        if (!latticezk_matvec(A, s, LATTICEZK_K, LATTICEZK_L, LATTICEZK_Q, result_ane)) {
            printf("  FAIL: ANE MatVec failed at iteration %d\n", t);
            r->cpu_consistency_fail++;
            continue;
        }

        // Compute via CPU (direct mod Q)
        cpu_matvec(A, s, LATTICEZK_K, LATTICEZK_L, LATTICEZK_Q, result_cpu);

        // Compare results
        int match = 1;
        for (int i = 0; i < LATTICEZK_K; i++) {
            if (result_ane[i] != result_cpu[i]) {
                match = 0;
                printf("  Mismatch at index %d: ANE=%llu, CPU=%llu\n",
                       i, (unsigned long long)result_ane[i], (unsigned long long)result_cpu[i]);
                break;
            }
        }

        if (match) {
            r->cpu_consistency_pass++;
        } else {
            r->cpu_consistency_fail++;
        }
    }

    printf("  Result: %d/%d matched (%.2f%%)\n\n",
           r->cpu_consistency_pass, r->cpu_consistency_pass + r->cpu_consistency_fail,
           100.0 * r->cpu_consistency_pass / (r->cpu_consistency_pass + r->cpu_consistency_fail));

    return r->cpu_consistency_fail == 0 ? 0 : 1;
}

int main(int argc, char *argv[]) {
    printf("Labrador Protocol Verification Suite\n");
    printf("=====================================\n\n");

    // Seed random for reproducibility (or use /dev/urandom in production)
    srand((unsigned int)time(NULL));

    TestResults results;
    init_results(&results);

    int failures = 0;

    // Run tests
    failures += test_completeness(&results);
    failures += test_soundness(&results);
    failures += test_ane_cpu_consistency(&results);

    // Summary
    printf("=== SUMMARY ===\n");
    printf("Completeness: %d/%d passed\n", results.completeness_pass, results.completeness_pass + results.completeness_fail);
    printf("Soundness: %d/%d correctly rejected\n", results.soundness_pass, results.soundness_pass + results.soundness_fail);
    printf("ANE/CPU Consistency: %d/%d matched\n", results.cpu_consistency_pass, results.cpu_consistency_pass + results.cpu_consistency_fail);

    if (failures == 0) {
        printf("\n✅ ALL TESTS PASSED - Protocol verified\n");
        return 0;
    } else {
        printf("\n❌ %d TEST(S) FAILED - Review needed\n", failures);
        return 1;
    }
}