#import "mil_cache.h"
#import "ane_runtime.h"
#import <CommonCrypto/CommonDigest.h>
#import <objc/runtime.h>

// T087: MIL program cache for zkML workloads.
//
// Cache key: SHA256(MIL text) → OrionProgram
//
// This cache solves the problem of repeated compilation for the same MIL
// with different weights (common in sumcheck, RNS, etc.).
//
// Thread-safe via @synchronized on the cache dictionary.
//
// IMPORTANT: Programs in cache are owned by the cache. Callers must NOT
// release programs returned from orion_mil_cache_get(). For patched copies,
// callers MUST release the patched copy (not the master).

#pragma mark - Cache Entry

@interface _OrionMILCacheEntry : NSObject
@property (nonatomic, assign) OrionProgram *master;  // Original compiled program
@property (nonatomic, copy)   NSString *milText;      // Original MIL text
@property (nonatomic, strong) NSDate *cachedAt;     // For LRU eviction
@end

@implementation _OrionMILCacheEntry
- (void)dealloc {
    if (_master) {
        orion_release_program(_master);
        _master = NULL;
    }
}
@end

#pragma mark - SHA256 Hash

static NSString* _sha256Key(const char *mil_text) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(mil_text, (CC_LONG)strlen(mil_text), hash);

    NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        [hex appendFormat:@"%02x", hash[i]];
    }
    return hex;
}

#pragma mark - Cache Storage

static NSMutableDictionary<NSString*, _OrionMILCacheEntry*> *_cache = nil;
static int _cache_hits = 0;
static int _cache_misses = 0;

static void _ensureCache(void) {
    if (!_cache) {
        _cache = [NSMutableDictionary dictionary];
    }
}

#pragma mark - Public API

OrionProgram* orion_mil_cache_get(
    const char* mil_text,
    NSDictionary* wdict,
    const char* tag
) {
    if (!mil_text) return NULL;

    @synchronized ([_OrionMILCacheEntry class]) {
        _ensureCache();

        NSString *key = _sha256Key(mil_text);
        _OrionMILCacheEntry *entry = _cache[key];

        if (entry) {
            _cache_hits++;
            entry.cachedAt = [NSDate date];
            return entry.master;  // Return cached (DO NOT release)
        }

        _cache_misses++;

        // Not cached - compile and store
        OrionProgram *prog = orion_compile_mil(mil_text, wdict, tag);
        if (!prog) return NULL;

        entry = [[_OrionMILCacheEntry alloc] init];
        entry.master = prog;
        entry.milText = @(mil_text);
        entry.cachedAt = [NSDate date];

        _cache[key] = entry;

        return prog;  // Caller must NOT release - cache owns it
    }
}

OrionProgram* orion_mil_cache_get_with_weights(
    const char* mil_text,
    NSDictionary* wdict,
    const char* tag
) {
    if (!mil_text) return NULL;

    @synchronized ([_OrionMILCacheEntry class]) {
        _ensureCache();

        NSString *key = _sha256Key(mil_text);
        _OrionMILCacheEntry *entry = _cache[key];

        if (!entry) {
            // No cached master - fall back to normal compilation
            return orion_compile_mil(mil_text, wdict, tag);
        }

        // Create patched copy of cached master
        // NOTE: orion_program_patch_weights does NOT increment compile_count
        // So this is essentially free (just file I/O + ANE load)
        OrionProgram *patched = orion_program_patch_weights(entry.master, mil_text, wdict, tag);

        return patched;  // Caller MUST release this
    }
}

void orion_mil_cache_store(
    const char* mil_text,
    OrionProgram* program
) {
    if (!mil_text || !program) return;

    @synchronized ([_OrionMILCacheEntry class]) {
        _ensureCache();

        NSString *key = _sha256Key(mil_text);

        // If already cached, entry dealloc will release old program
        _OrionMILCacheEntry *entry = [[_OrionMILCacheEntry alloc] init];
        entry.master = program;
        entry.milText = @(mil_text);
        entry.cachedAt = [NSDate date];

        _cache[key] = entry;
    }
}

bool orion_mil_cache_contains(const char* mil_text) {
    if (!mil_text) return false;

    @synchronized ([_OrionMILCacheEntry class]) {
        _ensureCache();
        NSString *key = _sha256Key(mil_text);
        return _cache[key] != nil;
    }
}

int orion_mil_cache_size(void) {
    @synchronized ([_OrionMILCacheEntry class]) {
        return _cache ? (int)_cache.count : 0;
    }
}

void orion_mil_cache_clear(void) {
    @synchronized ([_OrionMILCacheEntry class]) {
        if (_cache) {
            [_cache removeAllObjects];
        }
        _cache_hits = 0;
        _cache_misses = 0;
    }
}

void orion_mil_cache_evict(const char* mil_text) {
    if (!mil_text) return;

    @synchronized ([_OrionMILCacheEntry class]) {
        if (!_cache) return;
        NSString *key = _sha256Key(mil_text);
        [_cache removeObjectForKey:key];
    }
}

#pragma mark - Debug

int orion_mil_cache_stats(int *hits, int *misses) {
    @synchronized ([_OrionMILCacheEntry class]) {
        if (hits) *hits = _cache_hits;
        if (misses) *misses = _cache_misses;
        return _cache ? (int)_cache.count : 0;
    }
}
