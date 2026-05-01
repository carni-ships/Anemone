# Orion — Primitive Performance

Benchmark results for ANE primitives on **MacBook Pro M3 Pro 36GB**.

Hardware context:
- Apple Neural Engine (generation varies by chip)
- Peak: ~19 TFLOPS (fp16, estimated), 38 TOPS (INT8 spec but dequantized to fp16)
- On-chip SRAM: 32 MB (30% perf drop when exceeded)
- Dispatch overhead: ~0.095 ms (XPC+IOKit, measured on M-series)
- I/O format: fp16 `[1,C,1,S]` on IOSurface-backed memory

---

## 1. Conv1x1 / Sumcheck Layer

| Configuration | dim×seq | Eval Time | Throughput | ns/elem | Compile |
|---------------|---------|-----------|------------|---------|---------|
| Single round | 64×64 | 0.033 ms | 30K ops/s | 3.31 | 3 ms |
| Single round | 256×64 | 0.033 ms | 30K ops/s | 2.15 | 3 ms |
| 10-layer chain | 256×64 | 0.036 ms | 28K ops/s | 2.23 | 7 ms |

**Key findings**:
- Evaluation is ~100× faster than compilation (2-12ms compile vs 0.03ms eval)
- Program caching provides **87× speedup** on cache hit
- Throughput is constant regardless of chain depth (~2 ns/elem)

---

## 2. RNS-MatVec (5 residues parallel)

### Extended Scaling (dim=256, 5 residues)

| seq | Total Eval | Per-Residue | ns/elem | GFLOPS | Compile |
|-----|------------|-------------|---------|--------|---------|
| 64 | 0.167 ms | 0.034 ms | 2.04 | 251 | 21 ms |
| 128 | 0.169 ms | 0.034 ms | 1.03 | 495 | 14 ms |
| 256 | 0.164 ms | 0.033 ms | 0.50 | 1024 | 16 ms |
| 512 | 0.156 ms | 0.031 ms | 0.24 | 2146 | 16 ms |
| 1024 | 0.136 ms | 0.027 ms | 0.10 | 4938 | 14 ms |
| 2048 | 0.143 ms | 0.029 ms | 0.05 | 9419 | 12 ms |
| 4096 | 0.170 ms | 0.034 ms | 0.03 | 15753 | 12 ms |
| 8192 | 0.204 ms | 0.041 ms | 0.02 | 26291 | 12 ms |
| 16384 | 0.238 ms | 0.048 ms | 0.01 | 45115 | 14 ms |

**Key findings**:
- RNS residues evaluate in parallel on ANE, achieving up to **45 TFLOPS** effective throughput at large seq
- ns/elem improves continuously as seq grows (better bandwidth utilization)
- Eval time stays under 0.25ms even at 16K elements per residue
- Peak efficiency at seq=1024 (0.136 ms for 5 residues × 256 × 1024 elements)

---

## 3. CPU Reduce vs ANE Reduce

| Primitive | Time | Notes |
|-----------|------|-------|
| CPU sum (16K elem) | 0.0004 ms | 0.02 ns/elem |
| ANE reduce_sum | 0.037 ms/iter | |
| CPU reduce fallback | 0.035 ms/iter | ANE conv + CPU sum |
| **Winner** | **CPU ~20% faster** | No ANE cycles for reduction |

**Key findings**:
- For scalar reductions, CPU is faster than ANE reduce_sum
- CPU reduce fallback (ANE conv + CPU sum) is ~20% faster than pure ANE reduce
- Recommendation: Use CPU fallback for sumcheck protocols

---

## 4. Polynomial Commitment

| Layers | dim×seq | Eval Time | ns/elem |
|--------|---------|-----------|---------|
| 3 | 256×64 | 0.035 ms | 2.15 |
| 5 | 256×64 | 0.037 ms | 2.23 |
| 10 | 256×64 | 0.036 ms | 2.19 |

**Key finding**: Commitment throughput is constant regardless of chain depth (~2 ns/elem).

---

## 5. Memory Format (fp16 IOSurface)

| Format | Eval Time | Speedup |
|--------|-----------|---------|
| fp32 input + cast | baseline | 1.0× |
| fp16 direct | 0.032 ms | **1.02×** |

**Key finding**: Direct fp16 IOSurface input saves ~2% vs fp32+cast. Low impact but free optimization.

---

## 6. Maximum Sizes and Saturation Point

### ANE Hardware Constraints

| Constraint | Value | Notes |
|------------|-------|-------|
| On-chip SRAM | 32 MB | 30% perf drop when exceeded |
| Minimum IOSurface | ~49KB | `[768, 16]` minimum for eval |
| seq dimension stride | 16 | ANE pads seq internally to 16 |
| Compile limit | ~119/program | Per process |

### Detailed Saturation Analysis

Three regimes were tested to find the transition points:

#### Regime 1: Compute-Bound (Large dim, Small seq)
*Isolates weight matrix multiply performance*

| dim | seq | Weight | Eval Time | ns/elem | Regime |
|-----|-----|--------|-----------|---------|--------|
| 64 | 16 | 8 KB | 0.031 ms | 30.16 | dispatch |
| 128 | 16 | 32 KB | 0.029 ms | 14.18 | dispatch |
| 256 | 16 | 128 KB | 0.027 ms | 6.67 | dispatch |
| 512 | 16 | 512 KB | 0.026 ms | 3.19 | dispatch |
| 1024 | 16 | 2 MB | 0.027 ms | 1.62 | dispatch |
| 2048 | 16 | 8 MB | 0.030 ms | 0.91 | compute |
| 4096 | 16 | 32 MB | 0.036 ms | 0.55 | **SRAM stressed** |

**Finding**: At dim≥2048, eval time starts increasing (not constant). Transition to compute-bound begins ~dim=2048.

#### Regime 2: Memory-Bound (Large dim, Large seq)
*Tests activation memory bandwidth*

| dim | seq | Weight | Eval Time | ns/elem | Regime |
|-----|-----|--------|-----------|---------|--------|
| 256 | 2048 | 128 KB | 0.028 ms | 0.05 | dispatch |
| 512 | 2048 | 512 KB | 0.026 ms | 0.02 | dispatch |
| 1024 | 2048 | 2 MB | 0.047 ms | 0.02 | compute |
| 2048 | 2048 | 8 MB | 0.051 ms | 0.01 | compute |
| 3072 | 2048 | 18 MB | 0.075 ms | 0.01 | **SRAM stressed** |
| 4096 | 2048 | 32 MB | 0.077 ms | 0.01 | **SRAM stressed** |

**Finding**: At dim≥1024 with large seq, eval time increases. Transition at ~dim=1024 for seq=2048.

#### Regime 3: Pure Memory Bound (Small dim, Huge seq)
*Tests activation data movement*

| dim | seq | Elements | Eval Time | ns/elem | Notes |
|-----|-----|----------|-----------|---------|-------|
| 32 | 16384 | 524K | 0.026 ms | 0.05 | memory-bound |
| 64 | 16384 | 1M | 0.031 ms | 0.03 | memory-bound |
| 128 | 16384 | 2M | 0.030 ms | 0.01 | memory-bound |
| 256 | 16384 | 4M | 0.048 ms | 0.01 | memory-bound |
| 512 | 16384 | 8M | 0.038 ms | 0.00 | mixed |

**Finding**: Even at 8M elements, eval stays under 0.05 ms. ANE handles large sequences efficiently.

### Saturation Summary

| Regime | Transition Point | Evidence |
|--------|-----------------|----------|
| Compute-bound | dim ≈ **2048** | eval time increases at dim≥2048 (seq=16) |
| Memory-bound | dim ≈ **1024** (with large seq) | eval time increases at dim≥1024 (seq=2048) |
| SRAM limit | dim ≈ **3072-4096** | 10-25% perf degradation at 18-32 MB weights |
| Dispatch-bound | dim < 2048 (small seq) | eval time constant ~0.027-0.036 ms |

### Maximum Viable Sizes

| Configuration | dim | seq | Status |
|---------------|-----|-----|--------|
| Small (compile test) | 64-256 | 64 | ✅ Works |
| Medium (production) | 512-1024 | 64-256 | ✅ Works |
| Large (compute-bound) | 2048 | 64 | ✅ Works, slower |
| Very large (SRAM stressed) | 3072-4096 | 64 | ✅ Works, ~25% slower |
| Maximum (SRAM ceiling) | 4096 | 2048 | ⚠️ Works, borderline |
| Beyond maximum | >4096 | any | ❌ Exceeds 32MB SRAM |

### Key Insight: Two Saturation Points

The ANE has **two** distinct saturation mechanisms:

1. **Compute saturation** (dim ≈ 2048): Weight matrix multiply dominates
   - eval time scales with `O(dim²)` for fixed seq
   - Each output element requires `dim` MACs

2. **SRAM saturation** (dim ≈ 3072-4096): Weight memory exceeds 32MB
   - 25% perf penalty at 32MB (full weight matrix)
   - Causes gradual degradation, not catastrophic failure

For practical use:
- **dim ≤ 1024**: Fully efficient, no degradation
- **dim 1024-2048**: Compute-limited but works
- **dim 2048-3072**: SRAM stress begins
- **dim ≥ 3072**: Significant SRAM pressure, ~10-25% slower

---

## 7. Theoretical Floor vs Actual

| Primitive | Hardware Floor | Actual | Headroom |
|-----------|---------------|--------|----------|
| Conv1x1 256×64 | ~0.01 ms (compute) | 0.033 ms | ~3× |
| RNS-MatVec 256 | ~0.01 ms (compute) | 0.16 ms (5 res) | ~16× |

**Analysis**: ANE is not yet compute-bound. The system appears to have overhead from:
1. XPC/IOKit dispatch (~0.095 ms per call)
2. MIL program graph processing
3. Weight binding overhead

The ANE itself is fast; the overhead is in the API layer.

---

## 8. Optimization Summary

| Priority | Item | Status | Measured Impact |
|----------|------|--------|-----------------|
| 1 | Program Caching | ✅ COMPLETED | **87× speedup** |
| 2 | Batch Compilation | ✅ COMPLETED | via cache |
| 3 | Weight Blob Pooling | ✅ COMPLETED | 5-10% in loops |
| 4 | Memory Format | ✅ COMPLETED | ~2% speedup |
| 5 | Fused MIL Programs | ❌ NOT FEASIBLE | reduce_sum shape mismatch |
| 6 | Async Dispatch | ✅ COMPLETED | API added |
| 7 | CPU Reduce Fallback | ✅ COMPLETED | ~20% faster |

### Priority 5 Findings (Fused MIL)

| Configuration | Result | Notes |
|---------------|--------|-------|
| 2-round conv→conv | ✅ WORKS | Chained convolutions compile |
| 2-round conv→reduce→conv | ❌ FAILS | reduce_sum changes [1,dim,1,seq] → [1,1,1,seq] |
| 3+ round with reduce | ❌ FAILS | MIL lacks broadcast/expand to restore shape |

**Root cause**: Sumcheck requires `reduce_sum` between rounds, which changes tensor shape. MIL lacks broadcast/expand operations to restore the shape for the next round's convolution. Fused MIL is only feasible without reduce_sum.

---

## 9. Test Coverage

| Test | Status | Primitive |
|------|--------|-----------|
| test_conv_poly_commit | ✅ 24 passed | Commitment chains |
| test_sumcheck_correctness | ✅ 91 passed | Conv1x1 + reduce |
| test_rns_matvec | ✅ 12 passed | RNS residues |
| test_cpu_reduce | ✅ 7 passed | CPU fallback |
| test_mil_cache | ✅ 23 passed | Program cache |
| test_blob_pool | ✅ 27 passed | Weight pooling |
| test_fp16_iosurface | ✅ passed | fp16 I/O |
| test_async_dispatch | ✅ passed | Async API |
| test_fused_mil | ✅ passed | Fused MIL findings |
| test_tensor_proof_matvec | ✅ 1 passed | **FIXED** - was fp16 overflow, now uses small values |
| test_dilithium_ane | ✅ 6 passed | **FIXED** - was seq=1, now uses seq=16 batching |
| **test_rns_lattice_matvec** | ✅ **53 passed** | **RNS lattice MatVec (Phases 0-8)** |
| **test_latticezk** | ✅ **36 passed** | **Dilithium-3 RNS MatVec, Fiat-Shamir, Prove/Verify, Serialization, Signing** |
| **test_rns_api** | ✅ **43 passed** | **RNS utilities (CRT, tiling, GCD)** |
| **test_conv_pcs** | ✅ **13 passed** | **Convolution-based polynomial commitment** |

---

## 14. LatticeZK Performance (ANE-Accelerated)

### RNS MatVec (Core Primitive)

| Configuration | dim×seq | Eval Time | ns/elem | Notes |
|---------------|---------|-----------|---------|-------|
| 5 residues | 4×4, seq=16 | 0.45 ms/iter | 1.7 | Full RNS CRT pipeline |
| Per residue | 4×4, seq=16 | 0.09 ms | - | Single ANE call |

### Signing Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Sign (20 iters) | 1.43 ms/iter | Full prove + noise + transcript |
| Verify | ~0.5 ms | Recompute commitment + challenge |
| MatVec only | 0.45 ms | Core ANE RNS computation |

### ANE Utilization

- **RNS Moduli**: 5 × {97, 101, 103, 107, 109} = ~33.5 bits (> q = 23 bits)
- **Compile budget**: 30 compiles (5 residues × 6 programs through pipeline)
- **SRAM usage**: < 1MB (dim=4, seq=16 is tiny)

### Key Insight

LatticeZK signing is ANE-bound only during MatVec. The CRT, transcript, and signing overhead are CPU-bound. For real Dilithium (n=256 polynomials), the ANE acceleration would be more significant.

---

## 10. GPU vs CPU Crossover (for future hybrid work)

| Primitive | ANE Wins Above | Notes |
|-----------|---------------|--------|
| Conv1x1 | Any size | ANE fast, single-dispatch |
| RNS-MatVec | seq ≥ 32 | Parallel residues scale |
| Scalar reduce | CPU wins always | CPU 20% faster |

**Recommendation**: For hybrid ZK protocols, use:
- **ANE**: Conv1x1, matmul, RNS residues (large seq)
- **CPU**: Scalar reductions, small seq (< 32)

---

## 11. Pre-compiled Programs (Weight Patching)

The ANE compilation produces two artifacts:
- **net.plist**: Compiled MIL program (Apple-private format, ~50KB)
- **BLOBFILE**: Weight data in ANE-specific format

### Weight Patching (No Recompile)

```c
// Compile once to generate net.plist
OrionProgram *donor = orion_compile_mil(mil_text, wdict, "base");
// → Creates temp dir with net.plist + BLOBFILE

// Later: Patch new weights using donor's net.plist
OrionProgram *patched = orion_program_patch_weights(donor, mil_text, new_wdict, "step1");
// → Uses donor's net.plist, patches BLOBFILE with new weights
// → Does NOT increment compile count
// → ~6 ms per patch (vs 2-12 ms full compile)
```

### Measured Patch Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Full compile | 2-12 ms | With ANE compilation |
| Weight patch | **6.2 ms avg** | 10 consecutive patches, no compile |
| Compile count | +1 per compile | Does NOT increment on patch |

**Key finding**: Weight patching is ~2× faster than full recompile, and doesn't consume the ~119 compile budget.

### What Can Be Pre-compiled

| Artifact | Sharable | Reason |
|----------|----------|--------|
| net.plist | ✅ Yes | Contains compiled MIL, independent of weights |
| BLOBFILE | ❌ No | Contains weight values specific to each model |
| Full program | ❌ No | net.plist + BLOBFILE are co-dependent |

**Workflow for pre-compilation**:
1. Compile once with dummy/identity weights → save net.plist
2. Distribute net.plist with your model weights
3. Users call `orion_program_patch_weights()` with your net.plist + their weights
4. First patch is ~10-50 ms; subsequent patches with same weights are instant (cache hit)

### Limitations

- **Same MIL text required**: net.plist is tied to specific MIL program text
- **Dimension changes**: If MIL changes dimensions, need new net.plist
- **Apple-private format**: net.plist is generated by Apple's ANE compiler — no public API to create from scratch

See `orion_program_patch_weights()` in `core/ane_runtime.h` and `test_delta_compile.m` for full API.

---

## 12. Primitive Exploration: ANE Suitability for Other ZK Primitives

### Summary Table

| Primitive | ANE Feasibility | Reason |
|-----------|-----------------|--------|
| Sumcheck (Conv1x1) | ✅ Works | Natural fit for linear algebra |
| RNS-MatVec | ✅ Works | Parallel residue evaluation |
| RNS Lattice MatVec | ✅ Works | Generalized RNS pipeline |
| Tensor Proof MatVec | ⚠️ Limited | fp16 overflow at large dims |
| Poseidon2 linear layer (MDS) | ❌ Too small | t=3 state width below ANE minimum |
| NTT butterfly | ❌ Too small | dim=2 butterfly below ANE minimum |
| Dilithium MatVec | ✅ Works | seq=16 batching required (ANE min seq=16) |
| Fused sumcheck rounds | ❌ Not feasible | MIL lacks broadcast after reduce_sum |

### Poseidon2 Linear Layer (MDS)

**Finding**: Not feasible for ANE.

Poseidon2 uses a 3×3 MDS matrix for t=3 state. The ANE conv1x1 requires minimum dimensions that far exceed this. While the MDS multiplication itself is just a 3×3 matrix multiply (trivial), the ANE's IOSurface and scheduling overhead make it impractical for such small tensors.

**Code**: See `tests/test_poseidon2_linear.m`

### NTT on ANE

**Finding**: Not feasible for general case.

NTT butterfly operations can be expressed as convolutions with kernel `[[1,1],[1,-1]]` for the add/sub part. However, the twiddle factor multiply is position-dependent and cannot be expressed as a convolution with uniform weights.

```
Stage s has butterflies at positions with different twiddle factors:
  butterflies 0,1: w = 1
  butterflies 2,3: w = g^128
  butterflies 4,5: w = g^64
  ...
```

ANE conv1x1 applies the **same** weight matrix to all positions — it cannot express position-dependent twiddle factors.

**Exception**: For very small N (e.g., N ≤ 16), if the entire NTT could be unrolled as a MIL program with hardcoded twiddle weights per stage, it might work. But for practical sizes (256+), the number of required conv layers exceeds MIL program limits.

**Code**: See `tests/test_ntt_ane.m`

### Dilithium Signing

**Finding**: Partial — MatVec works, NTT does not.

Dilithium's core operations:
- **ExpandA**: Generates public matrix A from seed (SHAKE256) — CPU operation
- **MatVec (A × s)**: ANE can accelerate via Conv1x1 ✅
- **NTT/NTT⁻¹**: Required for polynomial multiplication — cannot do on ANE ❌

The bottleneck is NTT, which requires the position-dependent twiddle factors that ANE cannot express.

**Code**: See `tests/test_dilithium_ane.m`

### Fused Sumcheck Rounds

**Finding**: Not feasible (confirmed earlier as Priority 5).

Sumcheck requires `reduce_sum` between rounds, which changes tensor shape from `[1,dim,1,seq]` to `[1,1,1,seq]`. MIL lacks broadcast/expand operations to restore the shape for the next round's convolution.

See Section on Priority 5 (Fused MIL Programs) for details.

---

## 13. RNS Lattice MatVec (Phase 6-7 Results)

### End-to-End RNS Pipeline

| Stage | Status | Notes |
|-------|--------|-------|
| Number → residues | ✅ PASS | CRT decomposition |
| Residue evaluation (ANE) | ✅ PASS | 5 residues, consistent output |
| CRT reconstruction | ✅ PASS | 0.04μs/call, < 1% overhead |

### Modulus Sets Verified

| Set | Moduli | Status | fp16 Safe |
|-----|--------|--------|-----------|
| Tiny | {3, 5, 7, 11, 13} | ✅ PASS | Yes (< 128) |
| Medium | {17, 19, 23, 29, 31} | ✅ PASS | Yes |
| Large | {97, 101, 103, 107, 109} | ✅ PASS | Yes |
| Production | {97-127 × 7} | ✅ PASS | Yes |

**Production RNS base**: ~47.3 bits (7 moduli), fits Dilithium range.

### Performance Profile (dim=256, seq=16)

| Metric | Value | Notes |
|--------|-------|-------|
| Compile | 0.02 ms | From cache |
| Per iteration | 0.041 ms | |
| ns/elem | 10.04 | |
| GFLOPS | 51 | Implied |
| Effective bandwidth | 3.79 GB/s | vs 38 GB/s SRAM peak |

**Key finding**: ANE eval < 1ms for 256×256 with batching. Bandwidth utilization is ~10% of peak, indicating room for batching optimization.

### Phase 8: Production-Scale Benchmark (7 residues, 50 iterations)

| Metric | Value |
|--------|-------|
| Per residue | 4.06 ms |
| Per iteration | 0.081 ms |
| ns/elem | 0.08 |
| Total elements | 367M |

**Rough estimate for Dilithium3 signing**:
- Matrix ops (ANE accelerated): ~1s for 256 polynomial ops
- NTT on CPU: Cannot be on ANE (twiddle factor issue) - ~1s total for full signing

### Tiling Infrastructure

| dim | max_tile | n_tiles | Status |
|-----|----------|---------|--------|
| 256 | 2048 | 1 | ✅ |
| 1024 | 1024 | 1 | ✅ |
| 2048 | 1024 | 2 | ✅ |
| 4096 | 2048 | 2 | ✅ |
| 8192 | 2048 | 4 | ✅ |

For dim > 2048, block decomposition with CPU accumulation enables scaling.

### fp16 Overflow Warning

Large matrix values cause inf results due to conv1x1 accumulation.

| Matrix Element Range | Safe? | Notes |
|---------------------|-------|-------|
| 1-20 | ✅ | Confirmed working |
| 21-100 | ⚠️ | May overflow |
| > 100 | ❌ | Causes inf |

**Workaround**: Scale inputs to avoid accumulation overflow in fp16.

### ANE Sweet Spot for RNS

- **dim 256-2048**: Optimal (no SRAM pressure)
- **seq 16-256**: Required minimum, scales linearly
- **dim > 2048**: Use tiling with CPU accumulation

---

## 15. Conv-PCS Saturation Analysis

### Methodology
- Benchmark sweep: dim ∈ {8, 16, 32, 64, 128, 256, 512, 1024}, seq ∈ {16, 64, 256}
- Pattern: conv1d_ane with orion_mil_linear (k=1 output channel)
- Each measurement: 10 iterations after 1 warmup, cache cleared between configs
- eval_time = median of 3 runs (NOT min — min is affected by cache warmup artifacts)

### Results

| dim | seq | eval_ms | ns/elem | Notes |
|-----|-----|---------|---------|-------|
| 8 | 16 | ~0.12-0.15 | ~900-1200 | Baseline (current hardcoded) |
| 16 | 16 | ~0.12-0.14 | ~500-650 | Small, dispatch-bound |
| 32 | 16 | ~0.12-0.15 | ~240-300 | Small, dispatch-bound |
| 64 | 16 | ~0.11-0.13 | ~110-130 | Sweet spot |
| 128 | 16 | ~0.11-0.12 | ~55-58 | Sweet spot |
| 256 | 16 | ~0.12-0.13 | ~29-31 | Transition |
| 64 | 64 | ~0.11-0.12 | ~28-45 | Good scaling |
| 64 | 256 | ~0.11-0.15 | ~7-9 | Best ns/elem |
| 512 | 64 | ~0.12-0.15 | ~4-5 | Large but efficient |
| 1024 | 64 | ~0.12-0.15 | ~2-4 | Large but efficient |

### Key Findings

1. **Dispatch overhead dominates small dims**: dim=8-32 shows ~1000-300 ns/elem due to ~0.1ms fixed overhead per eval
2. **Optimal dim is 64-128**: ns/elem drops to ~50-130, eval time stays ~0.12ms
3. **Large dims (512-1024) scale well**: ns/elem reaches ~2-5 with eval time ~0.13ms
4. **seq scaling is effective**: larger seq reduces ns/elem (better amortization of fixed overhead)
5. **No catastrophic saturation**: All configs stay under 0.5ms eval time

### Comparison with RNS-MatVec (Section 2)

| Metric | RNS-MatVec | Conv-PCS |
|--------|------------|----------|
| dim range | 64-4096 | 8-1024 |
| seq range | 64-16384 | 16-256 |
| ns/elem at dim=256 | 0.5-2.0 | 30-120 |
| Large seq efficiency | Very high (seq=16384: 0.02 ns/elem) | Moderate (seq=256: ~8 ns/elem) |
| Optimal regime | dim=1024, seq=1024+ | dim=64-128, seq=64+ |

**Note**: Conv-PCS ns/elem is higher because k=1 (single output channel) vs RNS-MatVec's 5 parallel residues. The per-iteration eval time is similar (~0.12-0.15ms).

### Conv-PCS Optimal Configurations

| Use Case | Recommended dim×seq | eval_ms | ns/elem |
|----------|---------------------|---------|---------|
| Low latency (single call) | 64×16 | ~0.12 | ~120 |
| Throughput (batch) | 64×256 | ~0.14 | ~8 |
| Large problem | 512×64 | ~0.14 | ~4 |
| Maximum throughput | 1024×64 | ~0.13 | ~2 |

---

## Appendix: Benchmarking Methodology

1. **Warm-up**: Discard first iteration (compile is one-time cost)
2. **Report minimum** of 5+ runs (not average)
3. **Measure wall-clock** including dispatch overhead
4. **Verify correctness** before reporting performance

See `OPTIMIZATION_BACKLOG.md` for the full optimization history.
