// orion_rns.m — RNS Utilities Implementation

#import "orion_rns.h"
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
