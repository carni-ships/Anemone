# ANE Optimization Backlog

## Priority 1: Program Caching (HIGH IMPACT)
- **Problem**: Compilation costs 2-5ms vs ~0.035ms evaluation (50-100x overhead)
- **Solution**: Cache compiled OrionProgram by MIL signature
- **Approach**:
  - Hash MIL text + weight shapes → cache key
  - Store compiled programs in NSCache or custom hash map
  - Support weight updates via rebinding (not recompilation)
  - Add `orion_program_cache_get/set/evict` API
- **Expected gain**: 15-50x speedup for multi-round workloads

## Priority 2: Batch Program Compilation (MEDIUM IMPACT)
- **Problem**: Each sumcheck round compiles separately (~2ms each)
- **Solution**: Compile all round programs once, cache by round index
- **Approach**:
  - Pre-compile 16 round programs at init time
  - Store in array, access by index
  - Only recompile if MIL changes
- **Expected gain**: ~10x for 10-round sumcheck

## Priority 3: Weight Blob Pooling (MEDIUM IMPACT)
- **Problem**: Each test allocates/free's weight blobs repeatedly
- **Solution**: Reuse weight blobs across evaluations
- **Approach**:
  - Pool-based allocation for weight blobs
  - Cache recently used blobs by size/content hash
  - Reduce allocation overhead in tight loops
- **Expected gain**: ~5-10% improvement in tight loops

## Priority 4: Memory Format Optimization (MEDIUM IMPACT)
- **Problem**: fp32 IOSurface → fp16 ANE → fp32 IOSurface conversion overhead
- **Solution**: Investigate fp16 IOSurface support on ANE
- **Approach**:
  - Test ANE with fp16 IOSurface directly
  - Check if ANE supports fp16 I/O natively
  - Benchmark vs fp32→fp16→fp32 conversion
- **Expected gain**: 10-20% if conversion is eliminated

## Priority 5: Fused MIL Programs (NOT FEASIBLE)
- **Problem**: Per-round eval has overhead; chaining layers is efficient but round-to-round is not
- **Solution**: Single fused MIL program for entire sumcheck protocol
- **Approach**:
  - Encode all rounds as one MIL program with MUXes
  - GPU selects active round via input parameter
  - Eliminates per-round dispatch overhead
- **Findings**:
  - Multiple convolutions CAN be chained in one MIL program (2-round tested OK)
  - BUT reduce_sum changes tensor shape [1,dim,1,seq] → [1,1,1,seq]
  - MIL lacks broadcast/expand to restore shape for next round
  - MIL lacks MUX/select for conditional output routing
  - Sumcheck protocol requires reduce_sum between rounds → shape mismatch
- **Status**: Not feasible - would require MIL broadcast + MUX operations

## Priority 6: Async Dispatch (LOW IMPACT)
- **Problem**: Synchronous orion_eval blocks while ANE works
- **Solution**: Background dispatch with callbacks
- **Approach**:
  - Dispatch to background queue
  - Use completion handler / notification
  - Overlap CPU work with ANE evaluation
- **Expected gain**: Better CPU utilization in async pipelines

## Priority 7: Reduce_sum CPU Fallback (LOW IMPACT)
- **Problem**: reduce_sum adds overhead on ANE
- **Solution**: Compute scalar reductions on CPU
- **Approach**:
  - ANE outputs full tensor via conv
  - CPU computes reduce_sum on result
  - Save ANE cycles for parallelizable work
- **Expected gain**: ~15% improvement if ANE reduces fewer ops

## Status

| Priority | Item | Status |
|----------|------|--------|
| 1 | Program Caching | **COMPLETED** |
| 2 | Batch Compilation | **COMPLETED** (achieved via program cache) |
| 3 | Weight Blob Pooling | **COMPLETED** |
| 4 | Memory Format | **COMPLETED** (fp16 IOSurface works, ~2% speedup - low impact) |
| 5 | Fused MIL Programs | **NOT FEASIBLE** (ANE rejects chained convolutions, no MIL MUX) |
| 6 | Async Dispatch | **COMPLETED** (API added, no native ANE async support) |
| 7 | CPU Reduce Fallback | **COMPLETED** (CPU reduce ~20% faster than ANE reduce) |
| 8 | RNS Lattice MatVec | **COMPLETED** (Phases 0-7 verified) |

## Priority 8: RNS Lattice MatVec (COMPLETED)

- **Problem**: Extend ANE RNS-MatVec from toy sizes to practical cryptographic sizes
- **Solution**: 8-phase implementation with extended moduli, CRT, tiling, and Dilithium mapping
- **Approach**:
  - Phase 0: API verification with tiny moduli {3,5,7,11,13}
  - Phase 1: Extended modulus sets (medium {17-31}, large {97-109})
  - Phase 2: Generalized CRT reconstruction (0.04μs/call)
  - Phase 3: Tiling infrastructure for dim > 2048
  - Phase 4: Dilithium MatVec via orion_mil_linear (128×128 verified)
  - Phase 5: Production RNS base (~47.3 bits with 7 moduli)
  - Phase 6: End-to-end RNS pipeline
  - Phase 7: Performance profiling (0.041ms/iter at 256×256)
  - Phase 8: Production-scale benchmark (7 residues, 0.081ms/iter)
- **Results**:
  - **53 tests passed**
  - CRT overhead: < 1% of total time
  - fp16 overflow threshold: ~20 for matrix elements
  - ANE sweet spot: dim 256-2048, seq 16-256
  - Production benchmark: 0.081ms/iter, ns/elem = 0.08
- **Files**: `tests/test_rns_lattice_matvec.m`
- **Documentation**: `PERFORMANCE.md` Section 13
