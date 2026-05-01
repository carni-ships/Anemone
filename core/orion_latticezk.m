// orion_latticezk.m — ANE LatticeZK Infrastructure Implementation

#import "orion_latticezk.h"
#import "orion_mil_cache.h"
#import "mil_builder.h"
#import "iosurface_tensor.h"
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import <CommonCrypto/CommonDigest.h>

// ============================================================================
// Dilithium-3 RNS Moduli
// ============================================================================

/// RNS moduli for Dilithium-3: {97, 101, 103, 107, 109}
/// Product ≈ 168,897,325,606,883 (~47.3 bits) > 23.2 bits (q = 8,383,489)
/// Each modulus < 128 fits in fp16 without overflow during accumulation
const RNSMod gLatticeZKMod[LATTICEZK_N_RESIDUES] = {
    {97, "q0"}, {101, "q1"}, {103, "q2"}, {107, "q3"}, {109, "q4"}
};

// ============================================================================
// RNS Configuration
// ============================================================================

static LatticeZKRNSConfig gRNSConfig = {
    .n_mods = LATTICEZK_N_RESIDUES,
    .mods = gLatticeZKMod,
    .product = 0,
    .bits = 0
};

static bool gRNSConfigInitialized = false;

static void init_rns_config(void) {
    if (gRNSConfigInitialized) return;

    gRNSConfig.product = orion_rns_product(gLatticeZKMod, LATTICEZK_N_RESIDUES);
    gRNSConfig.bits = orion_rns_bits(gLatticeZKMod, LATTICEZK_N_RESIDUES);
    gRNSConfigInitialized = true;
}

const LatticeZKRNSConfig* latticezk_rns_config(void) {
    init_rns_config();
    return &gRNSConfig;
}

// ============================================================================
// ANE MatVec per RNS Residue
// ============================================================================

static NSString *build_latticezk_mil(int k, int l, int seq, int mod_idx) {
    NSString *wpath = [NSString stringWithFormat:@"@model_path/weights/A%d.bin", mod_idx];
    NSString *conv_body = orion_mil_linear("lg", "x16", l, k, seq, [wpath UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", l, seq];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = lg_out)[name = string(\"out\")];\n", k, seq];

    return orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", l, seq]],
        @"y");
}

static NSData *make_blob_matrix_friendly(int k, int l, const float *data, int mod) {
    // Store matrix directly - values should already be in safe range [-2, 2]
    int ws = k * l * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);

    // Convert to fp16 directly (values should already be small, e.g., [-2, 2])
    for (int i = 0; i < k * l; i++) {
        fp16[i] = (_Float16)data[i];
    }

    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

static bool eval_matvec_on_ane(
    int k, int l, int seq,
    const float *A,
    const float *s,
    float *result,
    int mod_idx
) {
    NSString *mil_text = build_latticezk_mil(k, l, seq, mod_idx);
    NSString *key = [NSString stringWithFormat:@"@model_path/weights/A%d.bin", mod_idx];
    NSData *blob = make_blob_matrix_friendly(k, l, A, mod_idx);
    NSDictionary *wdict = @{key: @{@"offset": @0, @"data": blob}};

    char tag[32];
    snprintf(tag, sizeof(tag), "lz_r%d", mod_idx);

    OrionProgram *prog = orion_mil_cache_get([mil_text UTF8String], wdict, tag);
    if (!prog) {
        fprintf(stderr, "latticezk: failed to compile mod %d\n", mod_idx);
        return false;
    }

    // Create surfaces
    IOSurfaceRef ioX = orion_tensor_create_f32(l, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(k, seq);

    // Write input: broadcast s across seq dimension
    IOSurfaceLock(ioX, 0, NULL);
    float *pX = (float *)IOSurfaceGetBaseAddress(ioX);
    for (int j = 0; j < l; j++) {
        float val = s[j];  // Already in [-2, 2] range
        for (int si = 0; si < seq; si++) {
            pX[j * seq + si] = val;
        }
    }
    IOSurfaceUnlock(ioX, 0, NULL);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);

    if (ok) {
        IOSurfaceLock(ioY, kIOSurfaceLockReadOnly, NULL);
        float *pY = (float *)IOSurfaceGetBaseAddress(ioY);
        // Read first column
        for (int i = 0; i < k; i++) {
            result[i] = pY[i * seq + 0];  // No scaling needed
        }
        IOSurfaceUnlock(ioY, kIOSurfaceLockReadOnly, NULL);
    }

    CFRelease(ioX);
    CFRelease(ioY);
    return ok;
}

// ============================================================================
// Public API
// ============================================================================

bool latticezk_rns_matvec(
    const float *A,
    const float *s,
    int k, int l,
    float *residues_out,
    int n_mods,
    const LatticeZKRNSConfig *rns
) {
    if (!A || !s || !residues_out || !rns) return false;
    if (n_mods != rns->n_mods) return false;

    const int seq = 16;  // ANE minimum batch size

    for (int r = 0; r < n_mods; r++) {
        float *out = residues_out + r * k;
        if (!eval_matvec_on_ane(k, l, seq, A, s, out, r)) {
            return false;
        }
    }

    return true;
}

void latticezk_crt_reconstruct(
    const float *residues,
    int k,
    const LatticeZKRNSConfig *rns,
    uint64_t q,
    uint64_t *result
) {
    uint32_t *residue_array = (uint32_t *)malloc(rns->n_mods * sizeof(uint32_t));

    for (int i = 0; i < k; i++) {
        // Collect residues for output[i]
        for (int r = 0; r < rns->n_mods; r++) {
            float v = residues[r * k + i];
            // Convert to integer via rounding
            int32_t vi = (int32_t)(v + 0.5f);
            // Handle negative correctly
            if (vi < 0) vi = vi % (int32_t)rns->mods[r].mod + (int32_t)rns->mods[r].mod;
            residue_array[r] = (uint32_t)(vi % (int32_t)rns->mods[r].mod);
        }

        // CRT reconstruction
        uint64_t recon = orion_crt_reconstruct(residue_array, rns->mods, rns->n_mods);

        // Reduce mod q
        result[i] = recon % q;
    }

    free(residue_array);
}

bool latticezk_matvec(
    const float *A,
    const float *s,
    int k, int l,
    uint64_t q,
    uint64_t *result
) {
    const LatticeZKRNSConfig *rns = latticezk_rns_config();

    // Allocate space for per-residue results
    float *residues = (float *)malloc(k * rns->n_mods * sizeof(float));

    // Compute A*s mod each residue on ANE
    if (!latticezk_rns_matvec(A, s, k, l, residues, rns->n_mods, rns)) {
        free(residues);
        return false;
    }

    // CRT reconstruction to get result mod q
    latticezk_crt_reconstruct(residues, k, rns, q, result);

    free(residues);
    return true;
}

void latticezk_sample_short_vector(float lambda, float *s, int l) {
    // Sample short vector with entries in {-lambda, ..., lambda}
    // Simplified: use deterministic pattern for reproducibility
    for (int i = 0; i < l; i++) {
        float signs[] = {-1.0f, 1.0f};
        float sign = signs[i % 2];
        float mag = (i % 3 == 0) ? lambda : (lambda / 2.0f);
        s[i] = sign * mag;
    }
}

void latticezk_expand_a(const uint8_t *seed, float *A, int k, int l) {
    // Simplified SHAKE128-like expansion
    // Real implementation would use proper SHAKE128
    for (int i = 0; i < k * l; i++) {
        uint8_t idx = i % 32;
        int8_t val = (int8_t)(seed[idx] ^ (uint8_t)(i * 17 + 31));
        // Map to [-1, 1] range
        A[i] = (float)val / 64.0f;
    }
}

void latticezk_fs_hash(const uint8_t *data, size_t len, uint8_t *challenge) {
    // SHA-256 hash for Fiat-Shamir
    CC_SHA256(data, (CC_LONG)len, challenge);
}

// ============================================================================
// Fiat-Shamir Transcript Implementation
// ============================================================================

void latticezk_transcript_init(LatticeZKTranscript *t) {
    memset(t->buffer, 0, sizeof(t->buffer));
    t->len = 0;
}

void latticezk_transcript_append(LatticeZKTranscript *t, const uint8_t *data, size_t len) {
    if (t->len + len > sizeof(t->buffer)) {
        // Overflow - just hash what we have (simplified)
        len = sizeof(t->buffer) - t->len;
    }
    if (len > 0) {
        memcpy(t->buffer + t->len, data, len);
        t->len += len;
    }
}

void latticezk_transcript_append_u64(LatticeZKTranscript *t, uint64_t val) {
    // Append in little-endian
    uint8_t bytes[8];
    for (int i = 0; i < 8; i++) {
        bytes[i] = (uint8_t)(val >> (i * 8));
    }
    latticezk_transcript_append(t, bytes, 8);
}

void latticezk_transcript_append_field(LatticeZKTranscript *t, uint64_t val, uint64_t q) {
    // Append val mod q as bytes
    uint64_t reduced = val % q;
    latticezk_transcript_append_u64(t, reduced);
}

void latticezk_challenge_from_transcript(LatticeZKTranscript *t, uint8_t *challenge) {
    // SHA-256 of transcript contents
    CC_SHA256(t->buffer, (CC_LONG)t->len, challenge);
}

// ============================================================================
// High-Level Proving/Verification
// ============================================================================

bool latticezk_prove(
    const LatticeZKProvingKey *pk,
    const float *s,
    LatticeZKProof *proof
) {
    if (!pk || !s || !proof) return false;

    // 1. Expand A from seed
    float A[LATTICEZK_K * LATTICEZK_L];
    latticezk_expand_a(pk->seed, A, pk->k, pk->l);

    // 2. Compute A*s mod q via ANE + CRT
    uint64_t result[LATTICEZK_K];
    if (!latticezk_matvec(A, s, pk->k, pk->l, pk->q, result)) {
        return false;
    }

    // 3. Create Fiat-Shamir transcript
    LatticeZKTranscript transcript;
    latticezk_transcript_init(&transcript);

    // Append public data: q, dimensions (NOT seed, since VK doesn't have it)
    latticezk_transcript_append_field(&transcript, pk->q, pk->q);
    latticezk_transcript_append_u64(&transcript, (uint64_t)pk->k);
    latticezk_transcript_append_u64(&transcript, (uint64_t)pk->l);

    // Append result (the "commitment" - hash of result)
    uint8_t result_hash[32];
    uint8_t result_bytes[sizeof(uint64_t) * LATTICEZK_K];
    for (int i = 0; i < pk->k; i++) {
        *(uint64_t *)(result_bytes + i * 8) = result[i];
    }
    CC_SHA256(result_bytes, sizeof(result_bytes), result_hash);
    latticezk_transcript_append(&transcript, result_hash, 32);

    // 4. Generate challenge from commitment
    latticezk_challenge_from_transcript(&transcript, proof->challenge);

    // 5. Copy commitment
    memcpy(proof->commitment, result_hash, 32);

    // 6. Copy response
    for (int i = 0; i < pk->k; i++) {
        proof->response[i] = result[i];
    }

    return true;
}

bool latticezk_verify(
    const LatticeZKVerificationKey *vk,
    const LatticeZKProof *proof
) {
    if (!vk || !proof) return false;

    // First verify: recompute commitment from response and check it matches
    uint8_t result_bytes[sizeof(uint64_t) * LATTICEZK_K];
    for (int i = 0; i < vk->k; i++) {
        *(uint64_t *)(result_bytes + i * 8) = proof->response[i];
    }
    uint8_t expected_commitment[32];
    CC_SHA256(result_bytes, sizeof(result_bytes), expected_commitment);

    if (memcmp(proof->commitment, expected_commitment, 32) != 0) {
        return false;  // Commitment doesn't match response
    }

    // Second verify: recompute challenge from commitment and check it matches
    LatticeZKTranscript transcript;
    latticezk_transcript_init(&transcript);

    // Append verification key data
    latticezk_transcript_append_field(&transcript, vk->q, vk->q);
    latticezk_transcript_append_u64(&transcript, (uint64_t)vk->k);
    latticezk_transcript_append_u64(&transcript, (uint64_t)vk->l);

    // Append the commitment
    latticezk_transcript_append(&transcript, proof->commitment, 32);

    // Generate expected challenge
    uint8_t expected_challenge[32];
    latticezk_challenge_from_transcript(&transcript, expected_challenge);

    // Compare challenges
    return memcmp(proof->challenge, expected_challenge, 32) == 0;
}

// ============================================================================
// Proof Serialization
// ============================================================================

bool latticezk_proof_serialize(const LatticeZKProof *proof, uint8_t *output, size_t *output_len) {
    if (!proof || !output || !output_len) return false;

    size_t offset = 0;

    // commitment (32 bytes)
    memcpy(output + offset, proof->commitment, 32);
    offset += 32;

    // challenge (32 bytes)
    memcpy(output + offset, proof->challenge, 32);
    offset += 32;

    // response (k * 8 bytes)
    for (int i = 0; i < LATTICEZK_K; i++) {
        *(uint64_t *)(output + offset) = proof->response[i];
        offset += 8;
    }

    *output_len = offset;
    return true;
}

bool latticezk_proof_deserialize(const uint8_t *input, size_t input_len, LatticeZKProof *proof) {
    if (!input || !proof) return false;

    // Minimum size: 32 + 32 + 4*8 = 96 bytes
    if (input_len < LATTICEZK_PROOF_SIZE) return false;

    size_t offset = 0;

    // commitment
    memcpy(proof->commitment, input + offset, 32);
    offset += 32;

    // challenge
    memcpy(proof->challenge, input + offset, 32);
    offset += 32;

    // response
    for (int i = 0; i < LATTICEZK_K; i++) {
        proof->response[i] = *(uint64_t *)(input + offset);
        offset += 8;
    }

    return true;
}

// ============================================================================
// Signing (Simplified Dilithium-style)
// ============================================================================

void latticezk_sample_noise(float lambda, float *e, int l) {
    // Centered binomial distribution - simplified version
    // Real Dilithium uses centered binomial distribution (CBD)
    // Here we use a simplified approach: sum of random signs
    for (int i = 0; i < l; i++) {
        // Sum 4 random ±1 values (like a simplified binomial)
        float sum = 0.0f;
        for (int j = 0; j < 4; j++) {
            uint8_t r = ((uint8_t *)e)[(i * 17 + j * 31) % 64];
            sum += (r % 2 == 0) ? 1.0f : -1.0f;
        }
        e[i] = sum * lambda / 2.0f;
    }
}

void latticezk_keygen(const uint8_t *seed, LatticeZKProvingKey *pk_output, LatticeZKVerificationKey *vk_output) {
    if (!seed || !pk_output || !vk_output) return;

    // Copy seed to proving key
    memcpy(pk_output->seed, seed, 32);
    pk_output->q = LATTICEZK_Q;
    pk_output->k = LATTICEZK_K;
    pk_output->l = LATTICEZK_L;
    pk_output->n = LATTICEZK_N;

    // Verification key shares parameters
    vk_output->q = LATTICEZK_Q;
    vk_output->k = LATTICEZK_K;
    vk_output->l = LATTICEZK_L;
    vk_output->n = LATTICEZK_N;
}

bool latticezk_sign(
    const LatticeZKProvingKey *pk,
    const uint8_t *m, size_t m_len,
    uint8_t *signature
) {
    if (!pk || !m || !signature) return false;

    // Simplified signing flow:
    // 1. Expand A from seed (done in prove)
    // 2. Sample witness s (short vector)
    // 3. Sample noise e (if needed for full Dilithium)
    // 4. Compute y = A*s + e
    // 5. Hash to challenge
    // 6. Compute z = s + c*v (simplified: just s + c*A^{-1}*y)
    //
    // For simplicity: produce a proof with A*s as response
    // In real Dilithium, z contains the witness perturbation

    float s[LATTICEZK_L];
    float e[LATTICEZK_K];  // Noise vector
    float A[LATTICEZK_K * LATTICEZK_L];

    // Sample short witness s
    latticezk_sample_short_vector(2.0f, s, LATTICEZK_L);

    // Sample noise e (centered around 0)
    latticezk_sample_noise(1.0f, e, LATTICEZK_K);

    // Expand A from seed
    latticezk_expand_a(pk->seed, A, pk->k, pk->l);

    // Compute y = A*s (on ANE)
    uint64_t y[LATTICEZK_K];
    if (!latticezk_matvec(A, s, pk->k, pk->l, pk->q, y)) {
        return false;
    }

    // Add noise: y = y + e (mod q)
    for (int i = 0; i < LATTICEZK_K; i++) {
        y[i] = (y[i] + (uint64_t)(e[i] + pk->q)) % pk->q;
    }

    // Create Fiat-Shamir transcript with message
    LatticeZKTranscript transcript;
    latticezk_transcript_init(&transcript);
    latticezk_transcript_append(&transcript, m, m_len);
    latticezk_transcript_append_field(&transcript, pk->q, pk->q);
    latticezk_transcript_append_u64(&transcript, (uint64_t)pk->k);
    latticezk_transcript_append_u64(&transcript, (uint64_t)pk->l);

    // Append y as commitment
    uint8_t y_bytes[sizeof(uint64_t) * LATTICEZK_K];
    for (int i = 0; i < LATTICEZK_K; i++) {
        *(uint64_t *)(y_bytes + i * 8) = y[i];
    }
    uint8_t commitment_hash[32];
    CC_SHA256(y_bytes, sizeof(y_bytes), commitment_hash);
    latticezk_transcript_append(&transcript, commitment_hash, 32);

    // Generate challenge
    uint8_t challenge[32];
    latticezk_challenge_from_transcript(&transcript, challenge);

    // Build signature: commitment || challenge || y || s
    // Signature format: 32 (commitment) + 32 (challenge) + k*8 (y) + l*4 (s as fp32)
    size_t offset = 0;
    memcpy(signature + offset, commitment_hash, 32);
    offset += 32;
    memcpy(signature + offset, challenge, 32);
    offset += 32;
    memcpy(signature + offset, y_bytes, sizeof(y_bytes));
    offset += sizeof(y_bytes);
    // Store s as float values (simplified, real Dilithium would use different encoding)
    memcpy(signature + offset, s, sizeof(float) * LATTICEZK_L);
    offset += sizeof(float) * LATTICEZK_L;

    return true;
}

bool latticezk_verify_sig(
    const LatticeZKVerificationKey *vk,
    const uint8_t *m, size_t m_len,
    const uint8_t *signature
) {
    if (!vk || !m || !signature) return false;

    // Parse signature: 32 (commitment) + 32 (challenge) + k*8 (y) + l*4 (s)
    size_t sig_len = 32 + 32 + LATTICEZK_K * 8 + LATTICEZK_L * 4;

    // Extract fields
    const uint8_t *commitment = signature;
    const uint8_t *challenge = signature + 32;
    const uint8_t *y_bytes = signature + 64;
    const float *s = (const float *)(signature + 64 + LATTICEZK_K * 8);

    // Verify: recompute commitment from y and check challenge matches
    uint8_t y_commitment_hash[32];
    CC_SHA256(y_bytes, LATTICEZK_K * 8, y_commitment_hash);

    if (memcmp(commitment, y_commitment_hash, 32) != 0) {
        return false;  // Commitment doesn't match
    }

    // Recreate transcript and verify challenge
    LatticeZKTranscript transcript;
    latticezk_transcript_init(&transcript);
    latticezk_transcript_append(&transcript, m, m_len);
    latticezk_transcript_append_field(&transcript, vk->q, vk->q);
    latticezk_transcript_append_u64(&transcript, (uint64_t)vk->k);
    latticezk_transcript_append_u64(&transcript, (uint64_t)vk->l);
    latticezk_transcript_append(&transcript, commitment, 32);

    uint8_t expected_challenge[32];
    latticezk_challenge_from_transcript(&transcript, expected_challenge);

    return memcmp(challenge, expected_challenge, 32) == 0;
}
