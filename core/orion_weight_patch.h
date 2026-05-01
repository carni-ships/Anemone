// orion_weight_patch.h — Weight Patching API for ANE Programs
//
// Provides a clean API for weight patching that:
//   1. Tracks compile count to avoid hitting the ~119 limit
//   2. Supports batch weight updates without recompilation
//   3. Works with the program cache for efficient reuse
//
// Usage:
//   // Initialize with a donor program (compiled once)
//   OrionWeightPatchCtx *ctx = orion_weight_patch_create(mil_text, donor_wdict, tag);
//
//   // Later: patch new weights (no recompile, no compile count increment)
//   OrionProgram *prog = orion_weight_patch_get(ctx, new_wdict, new_tag);
//
//   // Clean up
//   orion_weight_patch_destroy(ctx);

#ifndef ORION_WEIGHT_PATCH_H
#define ORION_WEIGHT_PATCH_H

#import "ane_runtime.h"

/// Opaque weight patching context
typedef struct OrionWeightPatchCtx OrionWeightPatchCtx;

/// Create a weight patching context from a compiled donor program.
/// The donor is kept as a master for patching subsequent weights.
/// @param mil_text   MIL program text (must be same for all patches)
/// @param wdict      Initial weight dictionary for the donor
/// @param tag        Debug tag for the donor program
/// @return Weight patching context, or NULL on failure
OrionWeightPatchCtx* orion_weight_patch_create(
    const char *mil_text,
    NSDictionary *wdict,
    const char *tag
);

/// Get a program with patched weights (no recompile).
/// @param ctx    Weight patching context
/// @param wdict  New weight dictionary
/// @param tag    Debug tag for the patched program
/// @return Patched program (caller must release), or NULL on failure
OrionProgram* orion_weight_patch_get(
    OrionWeightPatchCtx *ctx,
    NSDictionary *wdict,
    const char *tag
);

/// Check how many patches have been applied.
/// @param ctx  Weight patching context
/// @return Number of patches applied so far
int orion_weight_patch_count(OrionWeightPatchCtx *ctx);

/// Check if we're approaching the compile limit.
/// @param ctx  Weight patching context
/// @param warn_threshold  Warning threshold (e.g., 100 for ~119 limit)
/// @return true if approaching limit, false otherwise
bool orion_weight_patch_near_limit(
    OrionWeightPatchCtx *ctx,
    int warn_threshold
);

/// Destroy a weight patching context (releases donor program).
/// @param ctx  Weight patching context to destroy
void orion_weight_patch_destroy(OrionWeightPatchCtx *ctx);

#endif // ORION_WEIGHT_PATCH_H
