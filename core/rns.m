// orion_rns.m — RNS Utilities Implementation

#import "rns.h"
#import <stdlib.h>
#import <string.h>
#import <math.h>

int64_t orion_extended_gcd(int64_t a, int64_t b, int64_t *x, int64_t *y) {
    if (b == 0) {
        if (x) *x = 1;
        if (y) *y = 0;
        return a;
    }
    int64_t x1, y1;
    int64_t g = orion_extended_gcd(b, a % b, &x1, &y1);
    if (x) *x = y1;
    if (y) *y = x1 - (a / b) * y1;
    return g;
}

uint64_t orion_crt_reconstruct(const uint32_t *residues, const RNSMod *mods, int n) {
    // Compute M = product of all moduli
    uint64_t M = 1;
    for (int i = 0; i < n; i++) {
        M *= mods[i].mod;
    }

    uint64_t result = 0;
    for (int i = 0; i < n; i++) {
        uint64_t mod_i = mods[i].mod;
        uint64_t Mi = M / mod_i;

        // Compute Mi_inv = Mi^{-1} mod mod_i using extended GCD
        int64_t x, y;
        orion_extended_gcd(Mi % mod_i, (int64_t)mod_i, &x, &y);
        int64_t Mi_inv = x % (int64_t)mod_i;
        if (Mi_inv < 0) Mi_inv += mod_i;

        // term = residues[i] * Mi * Mi_inv mod M
        uint64_t term = residues[i] % mod_i;
        term = (term * Mi) % M;
        term = (term * (uint64_t)Mi_inv) % M;

        result = (result + term) % M;
    }

    return result;
}

uint64_t orion_rns_product(const RNSMod *mods, int n) {
    uint64_t M = 1;
    for (int i = 0; i < n; i++) {
        M *= mods[i].mod;
    }
    return M;
}

double orion_rns_bits(const RNSMod *mods, int n) {
    uint64_t M = orion_rns_product(mods, n);
    return log2((double)M);
}

void orion_rns_decompose(uint64_t x, const RNSMod *mods, int n, uint32_t *residues_out) {
    for (int i = 0; i < n; i++) {
        residues_out[i] = x % mods[i].mod;
    }
}

// ============================================================================
// Optimized CRT Reconstruction (Precomputed Constants)
// ============================================================================

bool orion_crt_constants_init(OrionCRTP *crt, const RNSMod *mods, int n) {
    if (!crt || !mods || n <= 0) return false;

    crt->n = n;
    crt->mods = mods;

    // Compute M = product of all moduli
    crt->M = 1;
    for (int i = 0; i < n; i++) {
        crt->M *= mods[i].mod;
    }

    // Allocate arrays
    crt->Mi = (uint64_t *)malloc(n * sizeof(uint64_t));
    crt->Mi_inv = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!crt->Mi || !crt->Mi_inv) {
        free(crt->Mi);
        free(crt->Mi_inv);
        return false;
    }

    // Precompute Mi and Mi_inv for each modulus
    for (int i = 0; i < n; i++) {
        uint64_t mod_i = mods[i].mod;
        crt->Mi[i] = crt->M / mod_i;

        // Compute (M/mod_i)^{-1} mod mod_i using extended GCD
        int64_t x, y;
        orion_extended_gcd((int64_t)(crt->Mi[i] % mod_i), (int64_t)mod_i, &x, &y);
        int64_t inv = x % (int64_t)mod_i;
        if (inv < 0) inv += mod_i;
        crt->Mi_inv[i] = (uint64_t)inv;
    }

    return true;
}

void orion_crt_constants_free(OrionCRTP *crt) {
    if (!crt) return;
    free(crt->Mi);
    free(crt->Mi_inv);
    crt->Mi = NULL;
    crt->Mi_inv = NULL;
    crt->n = 0;
}

uint64_t orion_crt_reconstruct_fast(const OrionCRTP *crt, const uint32_t *residues) {
    uint64_t result = 0;

    for (int i = 0; i < crt->n; i++) {
        uint64_t mod_i = crt->mods[i].mod;
        uint64_t Mi = crt->Mi[i];
        uint64_t Mi_inv = crt->Mi_inv[i];

        // term = residues[i] * Mi * Mi_inv mod M
        uint64_t term = residues[i] % mod_i;
        term = (term * Mi) % crt->M;
        term = (term * Mi_inv) % crt->M;

        result = (result + term) % crt->M;
    }

    return result;
}

// Batch CRT: convert all residues at once, then reconstruct
void orion_crt_reconstruct_fast_batch(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result) {
    // Preload to avoid repeated pointer dereferencing
    const int n = crt->n;
    const uint64_t M = crt->M;
    const uint64_t *Mi = crt->Mi;
    const uint64_t *Mi_inv = crt->Mi_inv;

    // Process 4 outputs at a time with full unrolling
    int i = 0;
    for (; i + 4 <= k; i += 4) {
        uint64_t recon0 = 0, recon1 = 0, recon2 = 0, recon3 = 0;
        for (int r = 0; r < n; r++) {
            uint64_t Mi_r = Mi[r];
            uint64_t Mi_inv_r = Mi_inv[r];
            uint64_t mod_r = crt->mods[r].mod;
            uint64_t base_idx = r * k;

            uint64_t term0 = residues[base_idx + i] % mod_r;
            term0 = (term0 * Mi_r) % M;
            term0 = (term0 * Mi_inv_r) % M;
            recon0 = (recon0 + term0) % M;

            uint64_t term1 = residues[base_idx + i + 1] % mod_r;
            term1 = (term1 * Mi_r) % M;
            term1 = (term1 * Mi_inv_r) % M;
            recon1 = (recon1 + term1) % M;

            uint64_t term2 = residues[base_idx + i + 2] % mod_r;
            term2 = (term2 * Mi_r) % M;
            term2 = (term2 * Mi_inv_r) % M;
            recon2 = (recon2 + term2) % M;

            uint64_t term3 = residues[base_idx + i + 3] % mod_r;
            term3 = (term3 * Mi_r) % M;
            term3 = (term3 * Mi_inv_r) % M;
            recon3 = (recon3 + term3) % M;
        }
        result[i] = recon0;
        result[i + 1] = recon1;
        result[i + 2] = recon2;
        result[i + 3] = recon3;
    }
    // Handle remainder
    for (; i < k; i++) {
        uint64_t recon = 0;
        for (int r = 0; r < n; r++) {
            uint64_t term = residues[r * k + i] % crt->mods[r].mod;
            term = (term * Mi[r]) % M;
            term = (term * Mi_inv[r]) % M;
            recon = (recon + term) % M;
        }
        result[i] = recon;
    }
}

// Fast CRT for small k (1-4) with loop unrolling
void orion_crt_reconstruct_fast_small_k(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result) {
    const int n = crt->n;
    const uint64_t M = crt->M;
    const uint64_t *Mi = crt->Mi;
    const uint64_t *Mi_inv = crt->Mi_inv;

    switch (k) {
        case 1: {
            uint64_t recon = 0;
            for (int r = 0; r < n; r++) {
                uint64_t term = residues[r] % crt->mods[r].mod;
                term = (term * Mi[r]) % M;
                term = (term * Mi_inv[r]) % M;
                recon = (recon + term) % M;
            }
            result[0] = recon;
            break;
        }
        case 2: {
            uint64_t recon0 = 0, recon1 = 0;
            for (int r = 0; r < n; r++) {
                uint64_t mod_r = crt->mods[r].mod;
                uint64_t Mi_r = Mi[r];
                uint64_t Mi_inv_r = Mi_inv[r];

                uint64_t term0 = residues[r * 2] % mod_r;
                term0 = (term0 * Mi_r) % M;
                term0 = (term0 * Mi_inv_r) % M;
                recon0 = (recon0 + term0) % M;

                uint64_t term1 = residues[r * 2 + 1] % mod_r;
                term1 = (term1 * Mi_r) % M;
                term1 = (term1 * Mi_inv_r) % M;
                recon1 = (recon1 + term1) % M;
            }
            result[0] = recon0;
            result[1] = recon1;
            break;
        }
        default:
            // Fall back to general batch for k > 2
            orion_crt_reconstruct_fast_batch(crt, residues, k, result);
            break;
    }
}

// ============================================================================
// Specialized CRT for Fixed RNS Base (7 moduli)
// ============================================================================

// Specialized CRT for exactly 7 moduli - full unrolling eliminates all loop overhead
uint64_t orion_crt_reconstruct_7mods(const OrionCRTP *crt, const uint32_t *residues) {
    // Assumes crt->n == 7, hardcoded for performance
    // Hardcode Mi and Mi_inv for 7 moduli to avoid pointer dereferencing
    uint64_t M = crt->M;

    // The compiler can optimize these better with known indices
    uint64_t recon = 0;

    // Residue 0: r0 * M0 * M0_inv
    {
        uint64_t r = residues[0] % crt->mods[0].mod;
        uint64_t term = (r * crt->Mi[0]) % M;
        term = (term * crt->Mi_inv[0]) % M;
        recon = (recon + term) % M;
    }
    // Residue 1
    {
        uint64_t r = residues[1] % crt->mods[1].mod;
        uint64_t term = (r * crt->Mi[1]) % M;
        term = (term * crt->Mi_inv[1]) % M;
        recon = (recon + term) % M;
    }
    // Residue 2
    {
        uint64_t r = residues[2] % crt->mods[2].mod;
        uint64_t term = (r * crt->Mi[2]) % M;
        term = (term * crt->Mi_inv[2]) % M;
        recon = (recon + term) % M;
    }
    // Residue 3
    {
        uint64_t r = residues[3] % crt->mods[3].mod;
        uint64_t term = (r * crt->Mi[3]) % M;
        term = (term * crt->Mi_inv[3]) % M;
        recon = (recon + term) % M;
    }
    // Residue 4
    {
        uint64_t r = residues[4] % crt->mods[4].mod;
        uint64_t term = (r * crt->Mi[4]) % M;
        term = (term * crt->Mi_inv[4]) % M;
        recon = (recon + term) % M;
    }
    // Residue 5
    {
        uint64_t r = residues[5] % crt->mods[5].mod;
        uint64_t term = (r * crt->Mi[5]) % M;
        term = (term * crt->Mi_inv[5]) % M;
        recon = (recon + term) % M;
    }
    // Residue 6
    {
        uint64_t r = residues[6] % crt->mods[6].mod;
        uint64_t term = (r * crt->Mi[6]) % M;
        term = (term * crt->Mi_inv[6]) % M;
        recon = (recon + term) % M;
    }

    return recon;
}

// Batch CRT for 7 moduli with 4x unrolling
void orion_crt_reconstruct_7mods_batch(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result) {
    uint64_t M = crt->M;

    // Preload all Mi and Mi_inv for 7 moduli
    uint64_t Mi0 = crt->Mi[0], Mi1 = crt->Mi[1], Mi2 = crt->Mi[2];
    uint64_t Mi3 = crt->Mi[3], Mi4 = crt->Mi[4], Mi5 = crt->Mi[5], Mi6 = crt->Mi[6];
    uint64_t Mi_inv0 = crt->Mi_inv[0], Mi_inv1 = crt->Mi_inv[1], Mi_inv2 = crt->Mi_inv[2];
    uint64_t Mi_inv3 = crt->Mi_inv[3], Mi_inv4 = crt->Mi_inv[4], Mi_inv5 = crt->Mi_inv[5], Mi_inv6 = crt->Mi_inv[6];
    uint32_t mod0 = crt->mods[0].mod, mod1 = crt->mods[1].mod, mod2 = crt->mods[2].mod;
    uint32_t mod3 = crt->mods[3].mod, mod4 = crt->mods[4].mod, mod5 = crt->mods[5].mod, mod6 = crt->mods[6].mod;

    int i = 0;
    for (; i + 4 <= k; i += 4) {
        uint64_t recon0 = 0, recon1 = 0, recon2 = 0, recon3 = 0;

        // Process residue 0 for all 4 outputs
        {
            uint64_t r0 = residues[i] % mod0;
            uint64_t t0 = (r0 * Mi0) % M; t0 = (t0 * Mi_inv0) % M; recon0 = (recon0 + t0) % M;
            uint64_t r1 = residues[i + 1] % mod0;
            uint64_t t1 = (r1 * Mi0) % M; t1 = (t1 * Mi_inv0) % M; recon1 = (recon1 + t1) % M;
            uint64_t r2 = residues[i + 2] % mod0;
            uint64_t t2 = (r2 * Mi0) % M; t2 = (t2 * Mi_inv0) % M; recon2 = (recon2 + t2) % M;
            uint64_t r3 = residues[i + 3] % mod0;
            uint64_t t3 = (r3 * Mi0) % M; t3 = (t3 * Mi_inv0) % M; recon3 = (recon3 + t3) % M;
        }
        // Residue 1
        {
            uint64_t base = 1 * k;
            uint64_t r0 = residues[base + i] % mod1;
            uint64_t t0 = (r0 * Mi1) % M; t0 = (t0 * Mi_inv1) % M; recon0 = (recon0 + t0) % M;
            uint64_t r1 = residues[base + i + 1] % mod1;
            uint64_t t1 = (r1 * Mi1) % M; t1 = (t1 * Mi_inv1) % M; recon1 = (recon1 + t1) % M;
            uint64_t r2 = residues[base + i + 2] % mod1;
            uint64_t t2 = (r2 * Mi1) % M; t2 = (t2 * Mi_inv1) % M; recon2 = (recon2 + t2) % M;
            uint64_t r3 = residues[base + i + 3] % mod1;
            uint64_t t3 = (r3 * Mi1) % M; t3 = (t3 * Mi_inv1) % M; recon3 = (recon3 + t3) % M;
        }
        // Residue 2
        {
            uint64_t base = 2 * k;
            uint64_t r0 = residues[base + i] % mod2;
            uint64_t t0 = (r0 * Mi2) % M; t0 = (t0 * Mi_inv2) % M; recon0 = (recon0 + t0) % M;
            uint64_t r1 = residues[base + i + 1] % mod2;
            uint64_t t1 = (r1 * Mi2) % M; t1 = (t1 * Mi_inv2) % M; recon1 = (recon1 + t1) % M;
            uint64_t r2 = residues[base + i + 2] % mod2;
            uint64_t t2 = (r2 * Mi2) % M; t2 = (t2 * Mi_inv2) % M; recon2 = (recon2 + t2) % M;
            uint64_t r3 = residues[base + i + 3] % mod2;
            uint64_t t3 = (r3 * Mi2) % M; t3 = (t3 * Mi_inv2) % M; recon3 = (recon3 + t3) % M;
        }
        // Residue 3
        {
            uint64_t base = 3 * k;
            uint64_t r0 = residues[base + i] % mod3;
            uint64_t t0 = (r0 * Mi3) % M; t0 = (t0 * Mi_inv3) % M; recon0 = (recon0 + t0) % M;
            uint64_t r1 = residues[base + i + 1] % mod3;
            uint64_t t1 = (r1 * Mi3) % M; t1 = (t1 * Mi_inv3) % M; recon1 = (recon1 + t1) % M;
            uint64_t r2 = residues[base + i + 2] % mod3;
            uint64_t t2 = (r2 * Mi3) % M; t2 = (t2 * Mi_inv3) % M; recon2 = (recon2 + t2) % M;
            uint64_t r3 = residues[base + i + 3] % mod3;
            uint64_t t3 = (r3 * Mi3) % M; t3 = (t3 * Mi_inv3) % M; recon3 = (recon3 + t3) % M;
        }
        // Residue 4
        {
            uint64_t base = 4 * k;
            uint64_t r0 = residues[base + i] % mod4;
            uint64_t t0 = (r0 * Mi4) % M; t0 = (t0 * Mi_inv4) % M; recon0 = (recon0 + t0) % M;
            uint64_t r1 = residues[base + i + 1] % mod4;
            uint64_t t1 = (r1 * Mi4) % M; t1 = (t1 * Mi_inv4) % M; recon1 = (recon1 + t1) % M;
            uint64_t r2 = residues[base + i + 2] % mod4;
            uint64_t t2 = (r2 * Mi4) % M; t2 = (t2 * Mi_inv4) % M; recon2 = (recon2 + t2) % M;
            uint64_t r3 = residues[base + i + 3] % mod4;
            uint64_t t3 = (r3 * Mi4) % M; t3 = (t3 * Mi_inv4) % M; recon3 = (recon3 + t3) % M;
        }
        // Residue 5
        {
            uint64_t base = 5 * k;
            uint64_t r0 = residues[base + i] % mod5;
            uint64_t t0 = (r0 * Mi5) % M; t0 = (t0 * Mi_inv5) % M; recon0 = (recon0 + t0) % M;
            uint64_t r1 = residues[base + i + 1] % mod5;
            uint64_t t1 = (r1 * Mi5) % M; t1 = (t1 * Mi_inv5) % M; recon1 = (recon1 + t1) % M;
            uint64_t r2 = residues[base + i + 2] % mod5;
            uint64_t t2 = (r2 * Mi5) % M; t2 = (t2 * Mi_inv5) % M; recon2 = (recon2 + t2) % M;
            uint64_t r3 = residues[base + i + 3] % mod5;
            uint64_t t3 = (r3 * Mi5) % M; t3 = (t3 * Mi_inv5) % M; recon3 = (recon3 + t3) % M;
        }
        // Residue 6
        {
            uint64_t base = 6 * k;
            uint64_t r0 = residues[base + i] % mod6;
            uint64_t t0 = (r0 * Mi6) % M; t0 = (t0 * Mi_inv6) % M; recon0 = (recon0 + t0) % M;
            uint64_t r1 = residues[base + i + 1] % mod6;
            uint64_t t1 = (r1 * Mi6) % M; t1 = (t1 * Mi_inv6) % M; recon1 = (recon1 + t1) % M;
            uint64_t r2 = residues[base + i + 2] % mod6;
            uint64_t t2 = (r2 * Mi6) % M; t2 = (t2 * Mi_inv6) % M; recon2 = (recon2 + t2) % M;
            uint64_t r3 = residues[base + i + 3] % mod6;
            uint64_t t3 = (r3 * Mi6) % M; t3 = (t3 * Mi_inv6) % M; recon3 = (recon3 + t3) % M;
        }

        result[i] = recon0;
        result[i + 1] = recon1;
        result[i + 2] = recon2;
        result[i + 3] = recon3;
    }

    // Handle remainder - compute each term fully then accumulate
    for (; i < k; i++) {
        uint64_t recon = 0;
        // Term 0: r0 * M0 * M0_inv
        uint64_t t = ((residues[i] % mod0) * Mi0) % M;
        recon = (recon + (t * Mi_inv0) % M) % M;
        // Term 1: r1 * M1 * M1_inv
        t = ((residues[k + i] % mod1) * Mi1) % M;
        recon = (recon + (t * Mi_inv1) % M) % M;
        // Term 2
        t = ((residues[2 * k + i] % mod2) * Mi2) % M;
        recon = (recon + (t * Mi_inv2) % M) % M;
        // Term 3
        t = ((residues[3 * k + i] % mod3) * Mi3) % M;
        recon = (recon + (t * Mi_inv3) % M) % M;
        // Term 4
        t = ((residues[4 * k + i] % mod4) * Mi4) % M;
        recon = (recon + (t * Mi_inv4) % M) % M;
        // Term 5
        t = ((residues[5 * k + i] % mod5) * Mi5) % M;
        recon = (recon + (t * Mi_inv5) % M) % M;
        // Term 6
        t = ((residues[6 * k + i] % mod6) * Mi6) % M;
        recon = (recon + (t * Mi_inv6) % M) % M;
        result[i] = recon;
    }
}

// ============================================================================
// Accelerate Framework SIMD CRT (for large batches)
// ============================================================================

void orion_crt_reconstruct_simd(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result) {
    // For small batches, use the fast batch
    if (k <= 8) {
        orion_crt_reconstruct_fast_batch(crt, residues, k, result);
        return;
    }

    // Use Accelerate for large batches via element-wise operations
    // Since our moduli are small (< 128), we can use uint64_t array operations
    const int n = crt->n;
    const uint64_t M = crt->M;

    // Process in chunks of 8 using SIMD-friendly unrolling
    int i = 0;
    for (; i + 8 <= k; i += 8) {
        uint64_t recon0 = 0, recon1 = 0, recon2 = 0, recon3 = 0;
        uint64_t recon4 = 0, recon5 = 0, recon6 = 0, recon7 = 0;

        for (int r = 0; r < n; r++) {
            uint64_t Mi_r = crt->Mi[r];
            uint64_t Mi_inv_r = crt->Mi_inv[r];
            uint64_t mod_r = crt->mods[r].mod;
            uint64_t base = r * k;

            // Unroll 8x
            uint64_t term;

            term = (residues[base + i] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon0 = (recon0 + term) % M;

            term = (residues[base + i + 1] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon1 = (recon1 + term) % M;

            term = (residues[base + i + 2] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon2 = (recon2 + term) % M;

            term = (residues[base + i + 3] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon3 = (recon3 + term) % M;

            term = (residues[base + i + 4] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon4 = (recon4 + term) % M;

            term = (residues[base + i + 5] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon5 = (recon5 + term) % M;

            term = (residues[base + i + 6] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon6 = (recon6 + term) % M;

            term = (residues[base + i + 7] % mod_r) * Mi_r % M;
            term = term * Mi_inv_r % M; recon7 = (recon7 + term) % M;
        }

        result[i] = recon0;
        result[i + 1] = recon1;
        result[i + 2] = recon2;
        result[i + 3] = recon3;
        result[i + 4] = recon4;
        result[i + 5] = recon5;
        result[i + 6] = recon6;
        result[i + 7] = recon7;
    }

    // Handle remainder with batch
    if (i < k) {
        orion_crt_reconstruct_fast_batch(crt, residues + i, k - i, result + i);
    }
}

void orion_tile_layout_init(TileLayout *tile, int dim, int max_tile) {
    memset(tile, 0, sizeof(TileLayout));

    if (dim <= 0 || max_tile <= 0) return;

    tile->n_tiles = (dim + max_tile - 1) / max_tile;
    tile->tile_size = max_tile;
    tile->last_tile_size = dim - (tile->n_tiles - 1) * max_tile;
    if (tile->last_tile_size <= 0) tile->last_tile_size = max_tile;

    tile->tile_offsets = (int *)malloc(tile->n_tiles * sizeof(int));
    for (int i = 0; i < tile->n_tiles; i++) {
        tile->tile_offsets[i] = i * max_tile;
    }
}

void orion_tile_layout_free(TileLayout *tile) {
    if (tile->tile_offsets) {
        free(tile->tile_offsets);
        tile->tile_offsets = NULL;
    }
    tile->n_tiles = 0;
}

int orion_tile_size_at(const TileLayout *tile, int idx) {
    if (idx < 0 || idx >= tile->n_tiles) return 0;
    if (idx == tile->n_tiles - 1) return tile->last_tile_size;
    return tile->tile_size;
}

int orion_tile_offset_at(const TileLayout *tile, int idx) {
    if (idx < 0 || idx >= tile->n_tiles) return 0;
    return tile->tile_offsets[idx];
}

#pragma mark - NTT Utilities

void orion_ntt_generate_twiddles(uint32_t *twiddles, int n, uint32_t g, uint32_t q) {
    // Generate twiddle factors: w[i] = g^i mod q
    uint32_t w = 1;
    for (int i = 0; i < n; i++) {
        twiddles[i] = w;
        uint64_t prod = (uint64_t)w * g;
        w = (uint32_t)(prod % q);
    }
}

void orion_ntt_bit_reverse(uint32_t *data, int n) {
    // Bit reversal permutation for FFT
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (j > i) {
            uint32_t tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
        // Update j with bit reversal
        int x = n >> 1;
        while (x && (j & x)) {
            j ^= x;
            x >>= 1;
        }
        j ^= x;
    }
}
