#import "orion_blob_pool.h"
#import <CommonCrypto/CommonDigest.h>

// T088: Weight blob pool for ANE programs.
//
// Reduces malloc/free churn by caching weight blobs by content hash.
//
// Thread-safe via @synchronized.

#pragma mark - SHA256 Hash

static NSString* _sha256Data(const void* data, size_t size) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, (CC_LONG)size, hash);

    NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        [hex appendFormat:@"%02x", hash[i]];
    }
    return hex;
}

#pragma mark - Pool Storage

// Pool: SHA256(content_hash) → NSData (retained by pool)
static NSMutableDictionary<NSString*, NSData*> *_pool = nil;
static int _pool_hits = 0;
static int _pool_misses = 0;

static void _ensurePool(void) {
    if (!_pool) {
        _pool = [NSMutableDictionary dictionary];
    }
}

#pragma mark - Public API

NSData* orion_blob_pool_get(const void* data, size_t size) {
    if (!data || size == 0) return NULL;

    @synchronized (_pool) {
        _ensurePool();

        NSString *hash = _sha256Data(data, size);
        NSData *existing = _pool[hash];

        if (existing) {
            _pool_hits++;
            return existing;
        }

        _pool_misses++;

        // Create new NSData and store in pool
        NSData *nsData = [NSData dataWithBytes:data length:size];
        _pool[hash] = nsData;

        return nsData;
    }
}

void orion_blob_pool_release(NSData* data) {
    // No-op: pool keeps the data, caller just stops using it
    (void)data;
}

void orion_blob_pool_stats(int *active, int *cached) {
    @synchronized (_pool) {
        _ensurePool();
        if (active) *active = (int)_pool.count;
        if (cached) *cached = (int)_pool.count;
    }
}

void orion_blob_pool_clear(void) {
    @synchronized (_pool) {
        if (_pool) {
            [_pool removeAllObjects];
        }
        _pool_hits = 0;
        _pool_misses = 0;
    }
}