// orion_rns.h — RNS (Residue Number System) Utilities for Lattice Crypto
//
// Provides generalized RNS operations for ANE-accelerated lattice crypto:
//   - CRT reconstruction (arbitrary coprime moduli)
//   - Tiling for large matrices (block decomposition)
//   - RNS decomposition/recomposition
//
// Usage:
//   // Define RNS moduli
//   RNSMod mods[] = {
//       {97, "q0"}, {101, "q1"}, {103, "q2"}, {107, "q3"}, {109, "q4"}
//   };
//
//   // Decompose a number into residues
//   uint32_t residues[5];
//   orion_rns_decompose(12345, mods, 5, residues);
//
//   // Reconstruct via CRT
//   uint64_t reconstructed = orion_crt_reconstruct(residues, mods, 5);
//
//   // Tile a large matrix for ANE
//   TileLayout tile;
//   orion_tile_layout_init(&tile, 4096, 2048);

#ifndef ORION_RNS_H
#define ORION_RNS_H

#import <stdbool.h>
#import <stdint.h>

/// RNS modulus descriptor
typedef struct {
    uint32_t mod;        // Modulus value (must be coprime with others)
    const char *name;     // Debug name (may be NULL)
} RNSMod;

/// Tile layout for block decomposition
typedef struct {
    int n_tiles;             // Number of tiles
    int tile_size;           // Size of each tile (except possibly last)
    int last_tile_size;       // Size of last tile
    int *tile_offsets;        // Starting offset for each tile (owned)
} TileLayout;

// ============================================================================
// CRT Reconstruction
// ============================================================================

/// Extended GCD for computing modular inverses
/// @param a First operand
/// @param b Second operand
/// @param x Output: x coefficient (may be NULL)
/// @param y Output: y coefficient (may be NULL)
/// @return GCD(a, b)
int64_t orion_extended_gcd(int64_t a, int64_t b, int64_t *x, int64_t *y);

/// CRT reconstruction for arbitrary coprime moduli
/// @param residues  Array of residues (one per modulus)
/// @param mods     Array of modulus descriptors
/// @param n        Number of moduli
/// @return Reconstructed number (result mod M where M = product of all moduli)
uint64_t orion_crt_reconstruct(const uint32_t *residues, const RNSMod *mods, int n);

/// Compute product of all moduli (M)
/// @param mods  Array of modulus descriptors
/// @param n     Number of moduli
/// @return M = product of all moduli
uint64_t orion_rns_product(const RNSMod *mods, int n);

/// Compute bit width of RNS base (log2 of product)
/// @param mods  Array of modulus descriptors
/// @param n     Number of moduli
/// @return Approximate bit width
double orion_rns_bits(const RNSMod *mods, int n);

// ============================================================================
// RNS Decomposition
// ============================================================================

/// Decompose a number into RNS residues
/// @param x     Number to decompose
/// @param mods  Array of modulus descriptors
/// @param n     Number of moduli
/// @param residues_out  Output: array of residues (caller allocates n elements)
void orion_rns_decompose(uint64_t x, const RNSMod *mods, int n, uint32_t *residues_out);

// ============================================================================
// Tile Layout
// ============================================================================

/// Initialize tile layout for block decomposition
/// @param tile      Output: tile layout structure
/// @param dim      Full dimension to partition
/// @param max_tile Maximum size of each tile
void orion_tile_layout_init(TileLayout *tile, int dim, int max_tile);

/// Free tile layout resources
/// @param tile  Tile layout to free
void orion_tile_layout_free(TileLayout *tile);

/// Get tile size at index
/// @param tile  Tile layout
/// @param idx  Tile index
/// @return Size of the specified tile
int orion_tile_size_at(const TileLayout *tile, int idx);

/// Get tile offset at index
/// @param tile  Tile layout
/// @param idx  Tile index
/// @return Offset of the specified tile
int orion_tile_offset_at(const TileLayout *tile, int idx);

// ============================================================================
// Optimized CRT Reconstruction (Precomputed Constants)
// ============================================================================

/// Precomputed constants for fast CRT reconstruction with fixed RNS base
/// Computes Mi = M/mod_i and Mi_inv = Mi^{-1} mod mod_i once at initialization
typedef struct {
    int n;                      // Number of moduli
    const RNSMod *mods;         // Moduli array (not owned)
    uint64_t M;                 // Product of all moduli
    uint64_t *Mi;              // M / mod_i for each i [n] (owned)
    uint64_t *Mi_inv;           // (M/mod_i)^{-1} mod mod_i for each i [n] (owned)
} OrionCRTP;

/// Initialize precomputed CRT constants for a given RNS base
/// @param crt     Output: precomputed constants structure
/// @param mods   Array of modulus descriptors
/// @param n      Number of moduli
/// @return true on success
bool orion_crt_constants_init(OrionCRTP *crt, const RNSMod *mods, int n);

/// Free precomputed CRT constants
/// @param crt  Constants structure to free
void orion_crt_constants_free(OrionCRTP *crt);

/// Fast CRT reconstruction using precomputed constants
/// @param crt       Precomputed constants (from orion_crt_constants_init)
/// @param residues  Array of residues (one per modulus)
/// @return Reconstructed number mod M
uint64_t orion_crt_reconstruct_fast(const OrionCRTP *crt, const uint32_t *residues);

/// Batch CRT reconstruction using precomputed constants (optimized for multiple outputs)
/// Processes multiple outputs at once for better cache locality
/// @param crt       Precomputed constants
/// @param residues  Array of residues [n_mods * k] (row-major)
/// @param k         Number of outputs
/// @param result    Output: k-element result array
void orion_crt_reconstruct_fast_batch(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result);

/// Fast CRT for small k (1-4) with loop unrolling
void orion_crt_reconstruct_fast_small_k(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result);

// ============================================================================
// Specialized CRT for Fixed RNS Base (7 moduli)
// ============================================================================

/// Specialized CRT for 7 moduli (Dilithium-3 default RNS base)
/// Full unrolling eliminates all loop overhead and pointer dereferencing
/// @param crt       Precomputed constants (must have n=7)
/// @param residues  Array of 7 residues
/// @return Reconstructed number mod M
uint64_t orion_crt_reconstruct_7mods(const OrionCRTP *crt, const uint32_t *residues);

/// Batch CRT for 7 moduli with 4x unrolling
void orion_crt_reconstruct_7mods_batch(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result);

// ============================================================================
// Accelerate Framework SIMD CRT (for large batches)
// ============================================================================

/// SIMD batch CRT using Accelerate framework for large k
/// Falls back to scalar for small k or when vDSP unavailable
/// @param crt       Precomputed constants
/// @param residues  Array of residues [n_mods * k]
/// @param k         Number of outputs
/// @param result    Output: k-element result array
void orion_crt_reconstruct_simd(const OrionCRTP *crt, const uint32_t *residues, int k, uint64_t *result);

// ============================================================================
// NTT Utilities
// ============================================================================

/// Generate twiddle factors for NTT of size n
/// @param twiddles  Output array [n]
/// @param n         Transform size
/// @param g         Primitive root of the field
/// @param q         Modulus (must be prime)
void orion_ntt_generate_twiddles(uint32_t *twiddles, int n, uint32_t g, uint32_t q);

/// Bit reversal permutation for in-place NTT
/// @param data  Data to permute
/// @param n     Size of data (must be power of 2)
void orion_ntt_bit_reverse(uint32_t *data, int n);

#endif // ORION_RNS_H
