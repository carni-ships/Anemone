#ifndef ORION_MIL_CACHE_H
#define ORION_MIL_CACHE_H

#import "ane_runtime.h"

/// Simple MIL program cache for zkML workloads.
///
/// Problem: Compilation costs 2-5ms vs ~0.035ms evaluation (50-100x overhead).
/// Solution: Cache compiled programs by MIL text hash, reuse with weight patching.
///
/// Usage:
///   // First time: compile and cache
///   OrionProgram *prog = orion_mil_cache_get(mil_text, wdict, tag);
///   if (!prog) {
///       prog = orion_compile_mil(mil_text, wdict, tag);
///       orion_mil_cache_store(mil_text, prog);
///   }
///   orion_eval(prog, ...);
///   // Do NOT release - cache owns it
///
///   // Later: get same MIL (even with different weights) - cache hit!
///   OrionProgram *prog2 = orion_mil_cache_get_same_mil(mil_text, wdict2, tag2);
///   // Returns patched copy, cache still owns original
///
/// For sumcheck: Pre-compile all round programs once:
///   for (int r = 0; r < n_rounds; r++) {
///       char mil_buf[256];
///       snprintf(mil_buf, sizeof(mil_buf), "...", challenges[r]);
///       OrionProgram *prog = orion_mil_cache_get(mil_buf, wdict, tag);
///       if (!prog) { prog = orion_compile_mil(...); orion_mil_cache_store(mil_buf, prog); }
///       // Cache hit on subsequent runs!
///   }

/// Get cached program by MIL text, creating if needed.
/// If cached program exists, returns it directly (DO NOT release).
/// If not cached, compiles using orion_compile_mil and caches the result.
/// @param mil_text  MIL program source text
/// @param wdict     Weight dictionary for compilation
/// @param tag       Program tag for debugging (may be NULL)
/// @return Cached or newly compiled program, or NULL on failure.
OrionProgram* orion_mil_cache_get(
    const char* mil_text,
    NSDictionary* wdict,
    const char* tag
);

/// Get cached program and apply new weights (patched copy).
/// If cached master exists, creates a patched copy via orion_program_patch_weights.
/// The patched copy is NOT cached - caller must release it.
/// Only the master (original weights) is cached.
/// @param mil_text      MIL program source text (must match cached master)
/// @param wdict         New weight dictionary (patched into copy)
/// @param tag           Program tag for debugging
/// @return Patched program copy, or NULL if no cached master found.
OrionProgram* orion_mil_cache_get_with_weights(
    const char* mil_text,
    NSDictionary* wdict,
    const char* tag
);

/// Store a compiled program in the cache (takes ownership).
/// @param mil_text  MIL program source text (cache key)
/// @param program   Compiled program to cache (cache takes ownership)
void orion_mil_cache_store(
    const char* mil_text,
    OrionProgram* program
);

/// Check if a MIL program is cached (without creating).
/// @param mil_text  MIL program source text
/// @return true if cached, false otherwise.
bool orion_mil_cache_contains(const char* mil_text);

/// Get number of cached programs.
int orion_mil_cache_size(void);

/// Clear all cached programs (releases via orion_release_program).
void orion_mil_cache_clear(void);

/// Evict a specific cached program by MIL text.
void orion_mil_cache_evict(const char* mil_text);

/// Get cache statistics.
/// @param hits     Output: number of cache hits (may be NULL)
/// @param misses   Output: number of cache misses (may be NULL)
/// @return Current cache size.
int orion_mil_cache_stats(int *hits, int *misses);

#endif // ORION_MIL_CACHE_H
