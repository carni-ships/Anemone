#ifndef ORION_BLOB_POOL_H
#define ORION_BLOB_POOL_H

#import <Foundation/Foundation.h>

/// Weight blob pool for reducing allocation overhead in tight loops.
///
/// Problem: Each test allocates/frees weight blobs repeatedly, causing malloc churn.
/// Solution: Cache weight blobs by content hash, reuse across evaluations.
///
/// Usage:
///   // Get a pooled blob (retained)
///   NSData *blob = orion_blob_pool_get(data, size);
///   // Use in weight dict
///   wdict[@"@model_path/weights/w.bin"] = @{@"offset": @0, @"data": blob};
///   // Release when done (pool may keep it alive)
///   orion_blob_pool_release(blob);

/// Get a pooled weight blob (or create new if not cached).
/// The returned NSData is retained. Release with orion_blob_pool_release().
/// Two blobs with identical content will return the same pooled NSData.
/// @param data   Raw weight data
/// @param size   Size in bytes
/// @return Pooled NSData (retained), or NULL on failure.
NSData* orion_blob_pool_get(const void* data, size_t size);

/// Release a pooled blob (decrements ref count in pool).
/// @param data  Data to release (from orion_blob_pool_get)
void orion_blob_pool_release(NSData* data);

/// Get current pool stats.
/// @param active   Output: number of active entries (may be NULL)
/// @param cached    Output: number of entries in cache (may be NULL)
void orion_blob_pool_stats(int *active, int *cached);

/// Clear all cached blobs.
/// @note Any live references to cleared blobs become invalid.
void orion_blob_pool_clear(void);

#endif // ORION_BLOB_POOL_H