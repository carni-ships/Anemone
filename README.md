# Anemone

**Apple Neural Engine primitives for lattice-based zero-knowledge proof systems.**

Hardware-accelerated MatVec operations for SNARK, STARK, and lattice cryptography on Apple Silicon.

```
┌─────────────────────────────────────────────────────────────┐
│           ANE Acceleration for ZK Proofs                    │
│                                                             │
│  MatVec (Labrador)  ·  Poseidon2 Hash  ·  NTT               │
│  ★ ~19 TFLOPS fp16 on M4 ANE ★                              │
└─────────────────────────────────────────────────────────────┘
```

---

## What is Anemone?

Anemone provides low-level ANE (Apple Neural Engine) primitives for accelerating lattice-based zero-knowledge proof systems. It was extracted from the Orion project and adapted for ZK-specific workloads.

**Why ANE for ZK?**

The ANE is a dedicated ML accelerator available on all Apple Silicon devices (~2 billion devices). While Apple exposes it only through CoreML (which is inference-only), ANE is actually general-purpose for any matrix-vector computation:

- **~19 TFLOPS fp16** on M4 chips
- **Dedicated silicon** — runs in parallel with CPU/GPU
- **Low power consumption** — efficient for constant workloads
- **Large SRAM** — 16MB on M4, capable of holding large matrices

For lattice-based SNARKs (like Labrador used in Crystalline-EVM), the dominant operation is **matrix-vector multiplication (MatVec)** over finite fields — exactly what ANE excels at.

---

## How It Works

```
┌────────────────────────────────────────────────────────────────┐
│                    Anemone Stack                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────-─┐ │
│  │              ZK Prover (Crystalline-EVM)                  │ │
│  │   Labrador SNARK  ·  NovaIVC  ·  Poseidon2 commitments    │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                  │
│  ┌──────────────────────────┴────────────────────────────────┐ │
│  │              Anemone Runtime (orion_backend)              │ │
│  │   orion_ane_init() · orion_compile_mil() · orion_eval()   │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                  │
│  ┌──────────────────────────┴────────────────────────────────┐ │
│  │              Apple Neural Engine                          │ │
│  │   _ANEClient · _ANECompiler · MIL IR                      │ │
│  │   IOSurface-backed fp16 MatVec                            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Key Operations Accelerated

| Operation | Description | Speedup |
|-----------|-------------|---------|
| **MatVec (Labrador)** | Matrix-vector multiply for SNARK witness generation | ~10-30x vs CPU |
| **Poseidon2 Hash** | Hash chain for Merkle commitments | ~5-10x vs CPU |
| **NTT** | Number Theoretic Transform for polynomial multiplication | ~3-5x vs CPU |

### Integration with Crystalline-EVM

Crystalline-EVM uses Anemone for:

1. **Poseidon2 hashing** — ANE-accelerated hash chains for bytecode and storage Merkle trees
2. **Labrador proving** — MatVec operations for SNARK witness generation
3. **NovaIVC folding** — LCCCS accumulation with ANE-accelerated poseidon operations

---

## Quick Start

```rust
// In Crystalline-EVM (lattice-evm crate)
use lattice_evm::prover::{Prover, ProverConfig};

let prover = Prover::new(ProverConfig::default())?;
println!("ANE available: {}", prover.ane_available());

// Prover automatically uses ANE for MatVec operations
```

### Building

```bash
# Requires macOS with Apple Silicon
# The orion_backend crate will automatically detect ANE availability

cargo build --package lattice-evm --release
```

---

## Architecture

### ANE Memory Layout

All ANE I/O uses `fp16 [1, C, 1, S]` on IOSurface-backed memory:

```
CPU: [S, D] ──transpose──> ANE: [1, D, 1, S] ──MatVec──> [1, D, 1, 1]
                                            │
                         ANE: [1, D, 1, S] <──result──┘
```

### Compilation Pipeline

1. **Rust wrapper** generates MIL (Metal Intermediate Language) text
2. **`orion_compile_mil()`** compiles MIL → ANE microcode (cached)
3. **`orion_eval()`** executes on ANE with IOSurface I/O

---

## Performance

Based on benchmarks in Crystalline-EVM:

| Mode | Execution | Total | Target |
|------|-----------|-------|--------|
| StateDiff | 25ms | **0.14s** | <12s |
| Minimal | ~6s | ~6.1s | <12s |
| Medium | ~4.3s | ~4.5s | <12s |
| Full | ~1.5s | ~1.6s | <12s |

**Per-opcode proving**: ~30ms per opcode with NovaIVC folding

The ANE acceleration makes the proving layer (~118ms for 8 batches) negligible compared to execution (~1.5s).

---

## Comparison with Other Accelerators

| Accelerator | Lattice ZK Support | Availability | Power Efficiency |
|------------|-------------------|--------------|------------------|
| **NVIDIA GPU** | Excellent (cuFFT, cuPoly) | Desktop only | High |
| **AMD GPU** | Good (ROCm) | Desktop only | High |
| **Apple ANE** | Good (Anemone) | All Apple Silicon | **Very High** |
| **CPU AVX2/AVX-512** | Moderate | Universal | Low |

**Anemone's niche**: Edge deployment on MacBooks, iPads, iPhones where GPU power is limited.

---

## Repository Structure

```
Anemone/
├── core/              # Low-level ANE runtime bindings
├── compiler/         # MIL IR compiler
├── kernels/          # ANE kernel implementations
├── experiments/       # Performance experiments
├── docs/              # ANE constraint documentation
└── tests/             # Test suite
```

---

## History

Anemone is an independent project that uses the same low-level ANE techniques as Orion for writing MIL kernels. We are not affiliated with Orion — we simply studied their ANE implementation to understand how to write efficient ANE kernels for ZK operations.

Key differences from Orion:
- **Focus**: ZK proving (MatVec, Poseidon2, NTT) vs LLM inference
- **Field**: Lattice finite field Q=8383489 vs float16
- **API**: Direct Rust bindings vs C CLI
- **Origin**: Independent implementation using Orion's techniques as reference

---

## References

- [Orion](https://github.com/carni-ships/Orion) — Original ANE runtime
- [maderix/ANE](https://github.com/maderix/ANE) — ANE reverse-engineering
- [Labrador](https://github.com/OrionSecurity/labrador) — Lattice SNARK protocol
- [Crystalline-EVM](https://github.com/carni-ships/Crystalline-EVM) — zkEVM using Anemone

---

*Anemone: a small, fast neural engine accelerator for lattice ZK proofs.*
