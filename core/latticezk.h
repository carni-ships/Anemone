// orion_latticezk.h — ANE LatticeZK Infrastructure
//
// ANE-accelerated lattice-based zk-SNARK proving system.
// Uses RNS decomposition for modular arithmetic + ANE for matrix ops.
//
// Core Flow:
//   1. RNS Decomposition: q → small coprime moduli
//   2. Per-residue evaluation: A*s mod q_i on ANE (batched 1×1 conv)
//   3. CRT Reconstruction: Combine residues → A*s mod q
//   4. Fiat-Shamir: Generate challenge from transcript
//   5. Fold/Verify: Standard lattice proof verification
//
// Dilithium-3 parameters (k=4, l=4, n=256, q=8383489):
//   - A: k×l matrix of polynomials in R_q^n
//   - s: l-vector of short polynomials
//   - A*s: k-vector result
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I . -I core \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m \
//     core/orion_mil_cache.m core/orion_rns.m \
//     core/orion_latticezk.m -c
//
// Run:
//   (use via test_latticezk.m)

#ifndef ORION_LATTICEZK_H
#define ORION_LATTICEZK_H

#import <stdbool.h>
#import <stdint.h>
#import "ane_runtime.h"
#import "rns.h"

// ============================================================================
// Dilithium-3 Parameters
// ============================================================================

/// Dilithium-3 security parameters
/// k = A matrix rows, l = A matrix cols, n = polynomial degree
#define LATTICEZK_K 4
#define LATTICEZK_L 4
#define LATTICEZK_N 256
#define LATTICEZK_Q 8383489

/// Challenge size in bytes (SHA-256 output)
#define LATTICEZK_CHALLENGE_BYTES 32

// ============================================================================
// Fiat-Shamir Transcript
// ============================================================================

/// Transcript state for Fiat-Shamir transform
/// Accumulates field elements, commitments, and challenges
typedef struct {
    uint8_t buffer[1024];  // Transcript accumulation buffer
    size_t len;            // Current length
} LatticeZKTranscript;

/// Initialize empty transcript
void latticezk_transcript_init(LatticeZKTranscript *t);

/// Append bytes to transcript
void latticezk_transcript_append(LatticeZKTranscript *t, const uint8_t *data, size_t len);

/// Append a uint64 to transcript
void latticezk_transcript_append_u64(LatticeZKTranscript *t, uint64_t val);

/// Append field element (mod q) to transcript
void latticezk_transcript_append_field(LatticeZKTranscript *t, uint64_t val, uint64_t q);

/// Generate challenge from transcript (SHA-256 based)
void latticezk_challenge_from_transcript(LatticeZKTranscript *t, uint8_t *challenge);

// ============================================================================
// Proving/Verification Keys (Simplified)
// ============================================================================

/// Proving key (public)
typedef struct {
    uint8_t seed[32];           // A matrix seed
    uint64_t q;                 // Modulus
    int k, l, n;                // Dimensions
} LatticeZKProvingKey;

/// Verification key (public)
typedef struct {
    uint64_t q;                 // Modulus
    int k, l, n;                // Dimensions
} LatticeZKVerificationKey;

// ============================================================================
// Proof Structure (Simplified Dilithium-style)
// ============================================================================

/// Proof structure for latticeZK
/// Simplified: commitment + challenge + response
#define LATTICEZK_PROOF_SIZE (32 + 32 + LATTICEZK_K * 8)

typedef struct {
    uint8_t commitment[32];     // Hash of A*s (first response)
    uint8_t challenge[32];     // Fiat-Shamir challenge
    uint64_t response[LATTICEZK_K];  // A*s mod q (for verification)
} LatticeZKProof;

// ============================================================================
// RNS Base for Dilithium
// ============================================================================

/// RNS moduli selected for Dilithium-3
/// Product ≈ 47.3 bits > 23.2 bits (q) with headroom for operations
/// Each modulus < 128 fits in fp16 without overflow
#define LATTICEZK_N_RESIDUES 5
extern const RNSMod gLatticeZKMod[LATTICEZK_N_RESIDUES];

// ============================================================================
// RNS Configuration
// ============================================================================

/// RNS configuration for lattice crypto
typedef struct {
    int n_mods;                 // Number of moduli
    const RNSMod *mods;         // Moduli array
    uint64_t product;          // M = product of all moduli
    double bits;               // Bit width of M
    OrionCRTP *crt;            // Precomputed CRT constants (owned)
} LatticeZKRNSConfig;

/// Initialize RNS config for Dilithium-3
/// @return RNS configuration (caller does not own, do not free)
const LatticeZKRNSConfig* latticezk_rns_config(void);

// ============================================================================
// MatVec with RNS
// ============================================================================

/// Compute A*s mod q for each RNS residue on ANE
/// @param A         k×l matrix (row-major, fp32)
/// @param s         l-element vector (fp32)
/// @param k         A matrix rows
/// @param l         A matrix cols
/// @param residues  Output: k-element result for each residue (k×n_mods total)
/// @param n_mods    Number of RNS moduli
/// @param rns        RNS configuration
/// @return true on success
bool latticezk_rns_matvec(
    const float *A,
    const float *s,
    int k, int l,
    float *residues_out,
    int n_mods,
    const LatticeZKRNSConfig *rns
);

/// Reconstruct full q result from RNS residues via CRT
/// @param residues   Per-residue results (k×n_mods)
/// @param k          Number of output elements
/// @param rns        RNS configuration
/// @param q          Modulus to reconstruct to
/// @param result     Output: k-element result mod q
void latticezk_crt_reconstruct(
    const float *residues,
    int k,
    const LatticeZKRNSConfig *rns,
    uint64_t q,
    uint64_t *result
);

/// Batch CRT reconstruction for better cache locality
/// Processes all k outputs at once with contiguous memory access
void latticezk_crt_reconstruct_batch(
    const float *residues,
    int k,
    const LatticeZKRNSConfig *rns,
    uint64_t q,
    uint64_t *result
);

// ============================================================================
// High-Level API
// ============================================================================

/// Single-shot A*s mod q computation using ANE + CRT
/// @param A       k×l matrix (row-major, fp32)
/// @param s       l-element vector (fp32)
/// @param k       A rows
/// @param l       A cols
/// @param q       Modulus
/// @param result  Output: k-element result mod q
/// @return true on success
bool latticezk_matvec(
    const float *A,
    const float *s,
    int k, int l,
    uint64_t q,
    uint64_t *result
);

/// Generate random short vector s (for testing/proving)
/// @param lambda   Small value bound (e.g., 2 for ±2)
/// @param s        Output: l-element short vector
/// @param l        Vector length
void latticezk_sample_short_vector(float lambda, float *s, int l);

/// Generate matrix A from seed (simulates SHAKE128 ExpandA)
/// @param seed     32-byte seed
/// @param A        Output: k×l matrix (row-major)
/// @param k        Matrix rows
/// @param l        Matrix cols
void latticezk_expand_a(const uint8_t *seed, float *A, int k, int l);

// ============================================================================
// Fiat-Shamir Transcript
// ============================================================================

/// Initialize empty transcript
/// @param t Transcript state (caller allocates)
void latticezk_transcript_init(LatticeZKTranscript *t);

/// Append bytes to transcript
/// @param t Transcript state
/// @param data Bytes to append
/// @param len Length of data
void latticezk_transcript_append(LatticeZKTranscript *t, const uint8_t *data, size_t len);

/// Append uint64 to transcript
/// @param t Transcript state
/// @param val Value to append
void latticezk_transcript_append_u64(LatticeZKTranscript *t, uint64_t val);

/// Append field element (mod q) to transcript
/// @param t Transcript state
/// @param val Field element value
/// @param q Modulus
void latticezk_transcript_append_field(LatticeZKTranscript *t, uint64_t val, uint64_t q);

/// Generate challenge from transcript
/// @param t Transcript state
/// @param challenge Output buffer (32 bytes)
void latticezk_challenge_from_transcript(LatticeZKTranscript *t, uint8_t *challenge);

// ============================================================================
// Proof Serialization
// ============================================================================

/// Serialize proof to bytes
/// @param proof Proof to serialize
/// @param output Output buffer (must be at least LATTICEZK_PROOF_SIZE bytes)
/// @return true on success
bool latticezk_proof_serialize(const LatticeZKProof *proof, uint8_t *output, size_t *output_len);

/// Deserialize proof from bytes
/// @param input Input buffer containing serialized proof
/// @param input_len Input length
/// @param proof Output proof structure
/// @return true on success
bool latticezk_proof_deserialize(const uint8_t *input, size_t input_len, LatticeZKProof *proof);

// ============================================================================
// High-Level Proving/Verification
// ============================================================================

/// Generate proof for witness s
/// @param pk Proving key (contains A seed)
/// @param s Witness vector (short, l elements)
/// @param proof Output proof structure
/// @return true on success
bool latticezk_prove(
    const LatticeZKProvingKey *pk,
    const float *s,
    LatticeZKProof *proof
);

/// Verify proof
/// @param vk Verification key
/// @param proof Proof to verify
/// @return true if proof is valid
bool latticezk_verify(
    const LatticeZKVerificationKey *vk,
    const LatticeZKProof *proof
);

// ============================================================================
// Signing (Simplified Dilithium-style)
// ============================================================================

/// Sample noise vector from centered binomial distribution
/// In real Dilithium this is the "b" noise - here simplified for testing
/// @param lambda Noise bound (typically 2-4 for testing)
/// @param e Output noise vector
/// @param l Vector length
void latticezk_sample_noise(float lambda, float *e, int l);

/// Generate signing keypair from seed
/// @param seed 32-byte seed
/// @param pk_output Output proving key
/// @param vk_output Output verification key
void latticezk_keygen(const uint8_t *seed, LatticeZKProvingKey *pk_output, LatticeZKVerificationKey *vk_output);

/// Sign a message (simplified Dilithium-style)
/// @param pk Proving key
/// @param m Message bytes
/// @param m_len Message length
/// @param signature Output signature
/// @return true on success (false on rejection, caller should retry)
bool latticezk_sign(
    const LatticeZKProvingKey *pk,
    const uint8_t *m, size_t m_len,
    uint8_t *signature
);

/// Verify signature
/// @param vk Verification key
/// @param m Message
/// @param m_len Message length
/// @param signature Signature bytes
/// @return true if signature is valid
bool latticezk_verify_sig(
    const LatticeZKVerificationKey *vk,
    const uint8_t *m, size_t m_len,
    const uint8_t *signature
);

#pragma mark - T024: Batch Polynomial Evaluation

/// Evaluate N polynomials at same point using Horner's method
/// @param coeffs     [n_polys, degree+1] row-major coefficients
/// @param x          [seq] values to evaluate at
/// @param n_polys    Number of polynomials
/// @param degree     Degree of each polynomial
/// @param results_out [n_polys, seq] output (row-major)
/// @param seq        Number of evaluation points
/// @return true on success
bool latticezk_batch_poly_eval(
    const float *coeffs,
    const float *x,
    int n_polys,
    int degree,
    float *results_out,
    int seq
);

#pragma mark - T025: Inner Product

/// Compute inner product <a,b> = sum(a_i * b_i)
/// @param a        First vector [n]
/// @param b        Second vector [n]
/// @param n        Vector length
/// @param seq      Sequence length for batching
/// @param result   Output: inner product result
/// @return true on success
bool latticezk_inner_product(
    const float *a,
    const float *b,
    int n,
    int seq,
    float *result
);

#pragma mark - T026: Batch MatVec (Matrix-Matrix Multiplication)

/// Batch matrix-matrix multiplication: C = A * B for multiple B vectors
/// A is k×l, B is l×m, output C is k×m
/// @param A        k×l matrix (row-major)
/// @param B        l×m matrix (row-major)
/// @param k        A rows
/// @param l        A cols / B rows
/// @param m        B cols
/// @param C_out    k×m output (row-major)
/// @param rns      RNS configuration for modular arithmetic
/// @return true on success
bool latticezk_batch_matvec(
    const float *A,
    const float *B,
    int k, int l, int m,
    float *C_out,
    const LatticeZKRNSConfig *rns
);

#pragma mark - T027: NTT (Number Theoretic Transform)

/// Forward NTT (ANE-accelerated for N <= 16, CPU for larger)
/// @param data  Input/output data (in-place)
/// @param n     Transform size
/// @param q     Modulus
/// @param g     Primitive root
/// @return true if ANE was used, false if fell back to CPU
bool orion_ntt_forward(uint32_t *data, int n, uint32_t q, uint32_t g);

/// Batch NTT for multiple small polynomials (N <= 16)
/// Batches n_polys polynomials together for ANE efficiency
bool orion_ntt_forward_batch(uint32_t *data, int n_polys, int n,
                               uint32_t q, uint32_t g);

/// RNS-decomposed NTT for large N
/// Decomposes into RNS residues, processes per residue, reconstructs via CRT
bool orion_ntt_forward_rns(const uint32_t *data_in, uint32_t *data_out,
                             int n, uint32_t q, uint32_t g,
                             const RNSMod *mods, int n_mods);

/// Best NTT implementation selector
/// N <= 16: batch ANE
/// N <= 256: hybrid (ANE butterfly + CPU twiddle)
/// N > 256: RNS decomposition
bool orion_ntt_forward_v2(uint32_t *data, int n, uint32_t q, uint32_t g);

#endif // ORION_LATTICEZK_H
