# Labrador SNARK Formal Verification Plan

**Date**: 2026-05-01
**Status**: Planning
**Target**: Anemone Labrador/ANE Implementation

---

## Executive Summary

This document outlines a formal verification plan for the Labrador SNARK implementation in Anemone. Labrador is a lattice-based SNARK using ANE-accelerated MatVec for the Ajtai commitment.

**Current State**: Implementation complete, tests disabled, no formal verification.
**Goal**: Prove correctness, soundness, and security properties before production use.

---

## 1. Background: Labrador SNARK Protocol

### 1.1 Protocol Overview

Labrador is a **lattice-based SNARK** using the Ajtai commitment scheme:

```
┌─────────────────────────────────────────────────────────────┐
│                  Labrador SNARK Protocol                     │
├─────────────────────────────────────────────────────────────┤
│  Setup:                                                     │
│    - Generate matrix A from seed (expansion)                │
│    - Public parameters: seed, verification key              │
│                                                             │
│  Prove(s, witness):                                        │
│    1. Compute commitment c = A·s mod q (ANE-accelerated)   │
│    2. Generate Fiat-Shamir transcript                      │
│    3. Challenge ch = SHA256(transcript)                    │
│    4. Response r = A·s mod q (short vector decomposition) │
│    5. Output proof = (c, r)                                │
│                                                             │
│  Verify(proof):                                            │
│    1. Recompute commitment from response                   │
│    2. Recompute challenge                                 │
│    3. Check challenge matches                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ANE Acceleration

The core operation is matrix-vector multiplication `A·s mod q`:
- **Dimensions**: K=4 output rows, L=256 lattice dimension
- **ANE**: 1,098 GFLOPS at dim=256
- **RNS**: 5-moduli decomposition {97, 101, 103, 107, 109}

### 1.3 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q | 8383489 | Prime modulus (Dilithium-3 field) |
| K | 4 | Matrix A row count |
| L | 256 | Matrix A column count (lattice dimension) |
| N | 256 | Polynomial degree |
| λ | 2 | Short vector bound |

---

## 2. Verification Goals

### 2.1 Protocol Properties

| Property | Definition | Target |
|----------|------------|--------|
| **Completeness** | Honest prover always produces accepting proof | Prove |
| **Soundness** | Cheating prover cannot fool verifier | Prove |
| **Zero-Knowledge** | Proof reveals nothing about witness | Prove |
| **Unique Response** | Same witness → deterministic proof | Verify |

### 2.2 Implementation Invariants

| Invariant | Location | Property |
|-----------|----------|----------|
| CRT reconstruction: no aliasing | `rns.m` | RNS product > Q |
| Response bounds: r[i] < Q | `latticezk_prove` | All response elements in range |
| Commitment binding | `latticezk_prove/commit` | A·s committed before challenge |
| ANE/CPU consistency | `lattice_ops.rs` | Fallback produces same result |

---

## 3. Critical Bugs Found

### 3.1 Transcript Buffer Overflow (CRITICAL)

**Location**: `core/latticezk.m` (was `orion_latticezk.m`)

```c
if (t->len + len > sizeof(t->buffer)) {
    len = sizeof(t->buffer) - t->len;  // SILENT TRUNCATION!
}
```

**Issue**: Silent truncation on overflow enables proof forgery.
**Fix**: Return error on overflow, or enlarge buffer.

### 3.2 Response Bounds Not Enforced

**Location**: `latticezk_prove()`

```c
// Response computed but NOT validated:
for (int i = 0; i < LATTICEZK_L; i++) {
    proof->response[i] = (uint32_t)(A_s[i] + 0.5f) % LATTICEZK_Q;
}
```

**Fix**: Add bounds check before serialization.

### 3.3 Weak Matrix Expansion

**Location**: `latticezk_expand_a()`

```c
int8_t val = (int8_t)(seed[idx] ^ (uint8_t)(i * 17 + 31));
A[i] = (float)val / 64.0f;
```

**Issue**: XOR+linear is NOT cryptographic. Attackers could recover seed.
**Fix**: Replace with SHAKE128-256.

---

## 4. Formal Verification Tasks

### 4.1 High Priority

1. **Transcript Overflow Proof**: Prove truncated data doesn't affect challenge
2. **Response Bounds Proof**: Prove r[i] < Q for all valid witnesses
3. **Matrix Uniformity Proof**: Prove A is indistinguishable from random under seed secrecy

### 4.2 Medium Priority

4. **CRT Correctness**: Prove RNS product > Q, no aliasing
5. **ANE/CPU Consistency**: Prove ANE path ≡ CPU path (mod Q)
6. **Fiat-Shamir Determinism**: Prove same input → same challenge

### 4.3 Security Properties

7. **Completeness Proof**: ∀(pk,witness), Verify(Prove) = ACCEPT
8. **Soundness Proof**: Verify(proof) = ACCEPT → prover knows witness
9. **ZK Proof**: Transcript leaks no witness info beyond commitment

---

## 5. Recommended Fixes

### 5.1 Critical (Before Production)

| Issue | Fix | Priority |
|-------|-----|----------|
| Transcript overflow | Return error, enlarge buffer | CRITICAL |
| Response bounds | Add check in prover | CRITICAL |
| Weak matrix expansion | Replace with SHAKE128-256 | HIGH |

### 5.2 Recommended

| Issue | Fix | Priority |
|-------|-----|----------|
| Deterministic seed | Use proof seed, not timestamp | MEDIUM |
| fp16 tolerance analysis | Document ANE fp16 bounds | MEDIUM |

---

## 6. Timeline

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| **1: Critical Fixes** | Overflow, bounds, matrix | 1 week |
| **2: Protocol Proofs** | Completeness, soundness, ZK | 4-6 weeks |
| **3: Implementation** | Rust verification, ANE/CPU | 2-3 weeks |
| **4: Integration** | E2E tests, fuzzing | 2 weeks |

**Total: ~9-14 weeks**

---

## 7. References

- **Dilithium**: [CRYPTO 2017](https://pq-crystals.org/dilithium/data/dilithium-specification-round3.pdf)
- **Labrador**: Lattice SNARK for zkEVM (internal)
- **Fiat-Shamir**: [FOCS 1986](https://ia.cr/2017/550)

---

*Document version: 1.0*
*Last updated: 2026-05-01*
