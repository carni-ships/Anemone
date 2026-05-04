// orion_gpu_ntt.h — GPU-accelerated NTT via Metal
//
// Provides GPU-accelerated Number Theoretic Transform for lattice-based ZK
// using Apple's Metal GPU (not ANE - ANE can't do position-dependent twiddles)
//
// Currently supports:
//   - Dilithium (q=8380417, N=256)
//   - Kyber (q=3329, N=256)
//
// Usage:
//   // Initialize GPU NTT engine
//   OrionGpuNtt *ntt = orion_gpu_ntt_create();
//   orion_gpu_ntt_init_dilithium(ntt, 512);  // 512 polynomials
//
//   // Forward NTT
//   orion_gpu_ntt_forward_dilithium(ntt, data, 512);
//
//   // Inverse NTT
//   orion_gpu_ntt_inverse_dilithium(ntt, data, 512);
//
//   orion_gpu_ntt_destroy(ntt);

#ifndef ORION_GPU_NTT_H
#define ORION_GPU_NTT_H

#import <stdint.h>
#import <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GPU NTT Context
// ============================================================================

typedef struct OrionGpuNtt OrionGpuNtt;

/// Create GPU NTT engine (returns NULL if no Metal GPU)
OrionGpuNtt* orion_gpu_ntt_create(void);

/// Destroy GPU NTT engine
void orion_gpu_ntt_destroy(OrionGpuNtt *ntt);

/// Check if GPU is available
bool orion_gpu_ntt_available(void);

/// Initialize for Dilithium NTT (q=8380417, N=256)
/// @param ntt GPU NTT engine
/// @param max_polys Maximum number of polynomials to batch
/// @return true on success
bool orion_gpu_ntt_init_dilithium(OrionGpuNtt *ntt, uint32_t max_polys);

/// Initialize for Kyber NTT (q=3329, N=256)
/// @param ntt GPU NTT engine
/// @param max_polys Maximum number of polynomials to batch
/// @return true on success
bool orion_gpu_ntt_init_kyber(OrionGpuNtt *ntt, uint32_t max_polys);

/// Forward NTT for Dilithium (q=8380417, N=256)
/// @param ntt GPU NTT engine (must be initialized with init_dilithium)
/// @param data In/Out: polynomial data (256 * num_polys uint32_t elements)
/// @param num_polys Number of polynomials to transform
/// @return true on success
bool orion_gpu_ntt_forward_dilithium(OrionGpuNtt *ntt, uint32_t *data, uint32_t num_polys);

/// Inverse NTT for Dilithium
/// @param ntt GPU NTT engine
/// @param data In/Out: polynomial data
/// @param num_polys Number of polynomials to transform
/// @return true on success
bool orion_gpu_ntt_inverse_dilithium(OrionGpuNtt *ntt, uint32_t *data, uint32_t num_polys);

/// Forward NTT for Kyber (q=3329, N=256)
bool orion_gpu_ntt_forward_kyber(OrionGpuNtt *ntt, uint16_t *data, uint32_t num_polys);

/// Inverse NTT for Kyber
bool orion_gpu_ntt_inverse_kyber(OrionGpuNtt *ntt, uint16_t *data, uint32_t num_polys);

// ============================================================================
// Synchronous (blocking) API - simpler but blocks caller
// ============================================================================

/// Convenience: forward + inverse NTT (round-trip, useful for testing)
bool orion_gpu_ntt_roundtrip_dilithium(OrionGpuNtt *ntt, uint32_t *data, uint32_t num_polys);

// ============================================================================
// Performance hints
// ============================================================================

/// GPU is faster with large batches; minimum recommended: 64 polynomials
#define ORION_GPU_NTT_MIN_BATCH 64

/// Maximum polynomials per dispatch (for memory constraints)
#define ORION_GPU_NTT_MAX_BATCH 16384

#ifdef __cplusplus
}
#endif

#endif // ORION_GPU_NTT_H