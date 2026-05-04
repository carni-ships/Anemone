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

    // For small moduli, we can use optimized inline math
    // Each term is small enough to multiply directly without overflow
    for (int i = 0; i < k; i++) {
        uint64_t recon = 0;
        for (int r = 0; r < n; r++) {
            uint64_t mod_r = crt->mods[r].mod;
            uint64_t residue = residues[r * k + i];

            // Inline multiplication - compiler can optimize this better
            // when values are known to be small
            uint64_t term = (residue * Mi[r]) % M;
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
