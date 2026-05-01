// orion_conv_pcs.h — Convolution-Based Polynomial Commitment Scheme
//
// ANE-accelerated polynomial commitment using convolution.
// Uses batched 1×1 convolutions for commit/open operations.
//
// Core Flow:
//   1. Commit: Convolve polynomial P with random kernel K → commitment C
//   2. Open: Prove evaluation at point via batched convolutions
//   3. Verify: Check commitment against evaluation + Merkle proof
//
// Conv-PCS is a linear-only commitment (no pairing required).
// Suitable for hyperplonk-style ZK protocols.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m core/orion_rns.m \
//     core/orion_conv_pcs.m -c
//
// Run:
//   (use via test_conv_pcs.m)

#ifndef ORION_CONV_PCS_H
#define ORION_CONV_PCS_H

#import <stdbool.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "rns.h"

// ============================================================================
// Conv-PCS Parameters
// ============================================================================

/// Maximum polynomial degree we can handle
#define CONV_PCS_MAX_DEGREE 1024

/// Number of random kernel elements (commitment security)
#define CONV_PCS_KERNEL_SIZE 32

/// Merkle tree depth (for binding)
#define CONV_PCS_MERKLE_DEPTH 8

/// Commitment size in bytes (hash + merkle root)
#define CONV_PCS_COMMITMENT_SIZE 64

// ============================================================================
// Commitment Structure
// ============================================================================

/// Conv-PCS commitment
/// Contains: polynomial commitment hash + Merkle tree root
typedef struct {
    uint8_t poly_commit[32];    // Hash of polynomial evaluation
    uint8_t merkle_root[32];     // Merkle root of commitment tree
} ConvPCSCommitment;

/// Opening proof for a polynomial evaluation
typedef struct {
    uint8_t evaluation_proof[32];  // Proof of correct evaluation
    uint8_t merkle_proof[32 * CONV_PCS_MERKLE_DEPTH];  // Merkle path
    uint32_t merkle_depth;          // Actual depth used
} ConvPCSProof;

// ============================================================================
// Kernel Generation
// ============================================================================

/// Generate random kernel for commitment
/// @param seed Random seed (32 bytes)
/// @param kernel Output kernel tensor (kernel_size elements)
/// @param kernel_size Number of kernel elements
void conv_pcs_generate_kernel(const uint8_t *seed, float *kernel, int kernel_size);

// ============================================================================
// Commitment Operations
// ============================================================================

/// Commit to polynomial P using random kernel K via ANE convolution
/// @param poly Polynomial P (degree+1 elements, fp32)
/// @param poly_degree Degree of polynomial
/// @param kernel Random kernel K
/// @param kernel_size Size of kernel
/// @param commitment Output commitment structure
/// @return true on success
bool conv_pcs_commit(
    const float *poly,
    int poly_degree,
    const float *kernel,
    int kernel_size,
    ConvPCSCommitment *commitment
);

/// Commit multiple polynomials (batched for efficiency)
/// @param polys Array of polynomial coefficients (fp32)
/// @param n_polys Number of polynomials
/// @param poly_degree Degree of each polynomial
/// @param kernel Random kernel
/// @param kernel_size Kernel size
/// @param commitments Output commitments (n_polys elements)
/// @return true on success
bool conv_pcs_commit_batched(
    const float *polys,
    int n_polys,
    int poly_degree,
    const float *kernel,
    int kernel_size,
    ConvPCSCommitment *commitments
);

// ============================================================================
// Opening/Verification
// ============================================================================

/// Generate opening proof for polynomial evaluation
/// @param poly Polynomial P
/// @param poly_degree Degree of polynomial
/// @param point Evaluation point (scalar)
/// @param evaluation P(point)
/// @param kernel Random kernel
/// @param kernel_size Kernel size
/// @param proof Output opening proof
/// @return true on success
bool conv_pcs_open(
    const float *poly,
    int poly_degree,
    float point,
    float evaluation,
    const float *kernel,
    int kernel_size,
    ConvPCSProof *proof
);

/// Verify opening proof (simplified - structural check only)
/// @param commitment Original commitment
/// @param point Evaluation point
/// @param evaluation Claimed evaluation P(point)
/// @param proof Opening proof
/// @return true if proof is valid (structural check)
bool conv_pcs_verify(
    const ConvPCSCommitment *commitment,
    float point,
    float evaluation,
    const ConvPCSProof *proof
);

/// Verify opening proof with explicit polynomial info
/// @param commitment Original commitment
/// @param point Evaluation point
/// @param evaluation Claimed evaluation P(point)
/// @param kernel Random kernel (for recomputing commitment)
/// @param kernel_size Kernel size
/// @param proof Opening proof
/// @return true if proof is valid
bool conv_pcs_verify_with_kernel(
    const ConvPCSCommitment *commitment,
    float point,
    float evaluation,
    const float *kernel,
    int kernel_size,
    const ConvPCSProof *proof
);

// ============================================================================
// Serialization
// ============================================================================

/// Serialize commitment to bytes
/// @param commitment Commitment to serialize
/// @param output Output buffer (64 bytes)
void conv_pcs_commit_serialize(const ConvPCSCommitment *commitment, uint8_t *output);

/// Deserialize commitment from bytes
/// @param input Input buffer (64 bytes)
/// @param commitment Output commitment
void conv_pcs_commit_deserialize(const uint8_t *input, ConvPCSCommitment *commitment);

/// Serialize proof to bytes
/// @param proof Proof to serialize
/// @param output Output buffer
/// @param output_len Output length (filled by function)
void conv_pcs_proof_serialize(const ConvPCSProof *proof, uint8_t *output, size_t *output_len);

/// Deserialize proof from bytes
/// @param input Input buffer
/// @param input_len Input length
/// @param proof Output proof
/// @return true on success
bool conv_pcs_proof_deserialize(const uint8_t *input, size_t input_len, ConvPCSProof *proof);

#endif // ORION_CONV_PCS_H