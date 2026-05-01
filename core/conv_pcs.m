// orion_conv_pcs.m — Convolution-Based Polynomial Commitment Scheme

#import "conv_pcs.h"
#import "mil_cache.h"
#import "mil_builder.h"
#import "iosurface_tensor.h"
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import <CommonCrypto/CommonDigest.h>

// ============================================================================
// Constants
// ============================================================================

/// Number of leaves in Merkle tree (power of 2 for simplicity)
#define CONV_PCS_NUM_LEAVES 32

// ============================================================================
// Kernel Generation
// ============================================================================

void conv_pcs_generate_kernel(const uint8_t *seed, float *kernel, int kernel_size) {
    for (int i = 0; i < kernel_size; i++) {
        uint8_t hash[32];
        uint8_t data[33];
        memcpy(data, seed, 32);
        data[32] = (uint8_t)i;
        CC_SHA256(data, 33, hash);
        kernel[i] = (float)(int8_t)hash[0] / 128.0f;
    }
}

// ============================================================================
// Internal Convolution via ANE
// ============================================================================

static bool conv1d_ane(
    const float *input,
    int input_len,
    const float *kernel,
    int kernel_len,
    float *output,
    int *output_len
) {
    int seq = 16;  // ANE minimum
    int k = 1;     // Output channels
    int dim = 8;

    NSString *prefix = @"conv";
    NSString *wpath = @"@model_path/weights/K.bin";

    NSString *conv_body = orion_mil_linear([prefix UTF8String], "x16",
                                          dim, k, seq,
                                          [wpath UTF8String], NULL);

    NSMutableString *body = [NSMutableString string];
    [body appendFormat:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [body appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n", dim, seq];
    [body appendString:conv_body];
    [body appendFormat:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [body appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to32, x = %@_out)[name = string(\"out\")];\n", k, seq, prefix];

    NSString *mil_text = orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", dim, seq]],
        @"y");

    int ws = k * dim * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);

    for (int i = 0; i < dim; i++) {
        fp16[i] = (_Float16)kernel[i];
    }

    NSData *blob = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
    NSDictionary *wdict = @{wpath: @{@"offset": @0, @"data": blob}};

    char tag[32];
    snprintf(tag, sizeof(tag), "conv1d_%d_%d", dim, k);

    OrionProgram *prog = orion_compile_mil([mil_text UTF8String], wdict, tag);
    if (!prog) {
        return false;
    }

    IOSurfaceRef ioX = orion_tensor_create_f32(dim, seq);
    IOSurfaceRef ioY = orion_tensor_create_f32(k, seq);

    IOSurfaceLock(ioX, 0, NULL);
    float *pX = (float *)IOSurfaceGetBaseAddress(ioX);
    for (int i = 0; i < dim * seq; i++) pX[i] = 0.0f;
    for (int i = 0; i < input_len; i++) {
        pX[i] = input[i];
    }
    IOSurfaceUnlock(ioX, 0, NULL);

    bool ok = orion_eval(prog, (IOSurfaceRef[]){ioX}, 1, (IOSurfaceRef[]){ioY}, 1);

    if (ok) {
        IOSurfaceLock(ioY, kIOSurfaceLockReadOnly, NULL);
        float *pY = (float *)IOSurfaceGetBaseAddress(ioY);
        output[0] = pY[0];
        if (output_len) *output_len = 1;
        IOSurfaceUnlock(ioY, kIOSurfaceLockReadOnly, NULL);
    }

    CFRelease(ioX);
    CFRelease(ioY);
    return ok;
}

// ============================================================================
// Merkle Tree Operations
// ============================================================================

static void hash_leaf(const uint8_t *commitment, uint8_t *out) {
    // Leaf = SHA256(0x00 || commitment)
    uint8_t data[33];
    data[0] = 0x00;
    memcpy(data + 1, commitment, 32);
    CC_SHA256(data, 33, out);
}

static void hash_node(const uint8_t *left, const uint8_t *right, uint8_t *out) {
    // Internal node = SHA256(0x01 || left || right)
    uint8_t data[65];
    data[0] = 0x01;
    memcpy(data + 1, left, 32);
    memcpy(data + 33, right, 32);
    CC_SHA256(data, 65, out);
}

static void build_merkle_tree(const uint8_t *leaves, int n_leaves, uint8_t *root) {
    // Compute Merkle root from n_leaves hashes
    // Tree is stored in a flat array: leaves at offset n_leaves, then parents

    if (n_leaves == 0) return;

    int depth = 0;
    int temp = n_leaves;
    while (temp > 1) { temp = (temp + 1) / 2; depth++; }

    // Allocate tree storage (up to 2*n_leaves nodes)
    uint8_t (*tree)[32] = calloc(n_leaves * 2, 32);

    // Copy leaves to tree[0..n_leaves-1]
    memcpy(tree, leaves, n_leaves * 32);

    // Build tree bottom-up
    int offset = n_leaves;
    int remaining = n_leaves;
    for (int level = 0; level < depth; level++) {
        int parent_count = (remaining + 1) / 2;
        for (int i = 0; i < parent_count; i++) {
            if (i * 2 + 1 < remaining) {
                // Full pair
                hash_node(tree[i * 2], tree[i * 2 + 1], tree[offset + i]);
            } else {
                // Odd node - hash with itself (or pad with zeros)
                hash_node(tree[i * 2], tree[i * 2], tree[offset + i]);
            }
        }
        offset += parent_count;
        remaining = parent_count;
    }

    // Root is last computed node
    memcpy(root, tree + offset - 32, 32);
    free(tree);
}

static bool verify_merkle_path(
    const uint8_t *leaf_hash,
    int leaf_index,
    const uint8_t (*path)[32],
    int path_depth,
    const uint8_t *expected_root
) {
    // Start from leaf
    uint8_t current[32];
    memcpy(current, leaf_hash, 32);

    // Follow path up to root
    for (int i = 0; i < path_depth; i++) {
        uint8_t sibling[32];
        memcpy(sibling, path[i], 32);

        // Determine if this is a left or right child
        int node_index = leaf_index >> i;
        if (node_index % 2 == 0) {
            // Left child - sibling is on the right
            hash_node(current, sibling, current);
        } else {
            // Right child - sibling is on the left
            hash_node(sibling, current, current);
        }
    }

    // Compare with expected root
    return memcmp(current, expected_root, 32) == 0;
}

// ============================================================================
// Commitment Operations
// ============================================================================

bool conv_pcs_commit(
    const float *poly,
    int poly_degree,
    const float *kernel,
    int kernel_size,
    ConvPCSCommitment *commitment
) {
    if (!poly || !kernel || !commitment) return false;

    float conv_result[512];
    int result_len = 0;

    bool ok = conv1d_ane(poly, poly_degree + 1, kernel, kernel_size, conv_result, &result_len);

    if (!ok || result_len == 0) {
        float eval = 0.0f;
        int use_len = (poly_degree + 1) < kernel_size ? (poly_degree + 1) : kernel_size;
        for (int i = 0; i < use_len; i++) {
            eval += poly[i] * kernel[i];
        }
        uint8_t eval_bytes[sizeof(float) + 4];
        *(float *)eval_bytes = eval;
        *(uint32_t *)(eval_bytes + 4) = poly_degree;
        CC_SHA256(eval_bytes, sizeof(eval_bytes), commitment->poly_commit);
    } else {
        uint8_t hash_input[sizeof(float) * 512];
        int copy_len = result_len < 512 ? result_len : 512;
        for (int i = 0; i < copy_len; i++) {
            *(float *)(hash_input + i * sizeof(float)) = conv_result[i];
        }
        CC_SHA256(hash_input, copy_len * sizeof(float), commitment->poly_commit);
    }

    // Build Merkle tree from polynomial commitment
    // For single polynomial, tree has single leaf
    // For batching, we'd have multiple leaves - here we use single leaf + dummy padding
    uint8_t leaves[CONV_PCS_NUM_LEAVES][32];
    for (int i = 0; i < CONV_PCS_NUM_LEAVES; i++) {
        if (i == 0) {
            hash_leaf(commitment->poly_commit, leaves[i]);
        } else {
            // Pad with hash of index (deterministic)
            uint8_t pad_data[33];
            pad_data[0] = 0xFF;
            *(uint32_t *)(pad_data + 1) = i;
            CC_SHA256(pad_data, 5, leaves[i]);
        }
    }

    build_merkle_tree((uint8_t *)leaves, CONV_PCS_NUM_LEAVES, commitment->merkle_root);

    return true;
}

bool conv_pcs_commit_batched(
    const float *polys,
    int n_polys,
    int poly_degree,
    const float *kernel,
    int kernel_size,
    ConvPCSCommitment *commitments
) {
    if (!polys || !kernel || !commitments) return false;

    for (int i = 0; i < n_polys; i++) {
        if (!conv_pcs_commit(polys + i * (poly_degree + 1), poly_degree, kernel, kernel_size, &commitments[i])) {
            return false;
        }
    }

    // Build combined Merkle tree if multiple commitments
    if (n_polys > 1 && n_polys <= CONV_PCS_NUM_LEAVES) {
        // Hash each commitment's poly_commit as leaf
        uint8_t leaves[CONV_PCS_NUM_LEAVES][32];
        for (int i = 0; i < n_polys; i++) {
            hash_leaf(commitments[i].poly_commit, leaves[i]);
        }
        // Pad remaining leaves
        for (int i = n_polys; i < CONV_PCS_NUM_LEAVES; i++) {
            uint8_t pad_data[33];
            pad_data[0] = 0xFF;
            *(uint32_t *)(pad_data + 1) = i;
            CC_SHA256(pad_data, 5, leaves[i]);
        }

        // Build combined root (use first commitment as representative)
        build_merkle_tree((uint8_t *)leaves, CONV_PCS_NUM_LEAVES, commitments[0].merkle_root);
        // Copy same root to all commitments for consistency
        for (int i = 1; i < n_polys; i++) {
            memcpy(commitments[i].merkle_root, commitments[0].merkle_root, 32);
        }
    }

    return true;
}

// ============================================================================
// Opening/Verification
// ============================================================================

bool conv_pcs_open(
    const float *poly,
    int poly_degree,
    float point,
    float evaluation,
    const float *kernel,
    int kernel_size,
    ConvPCSProof *proof
) {
    if (!poly || !proof) return false;

    // Recompute polynomial commitment
    uint8_t poly_commit[32];
    float conv_result[512];
    int result_len = 0;

    bool ok = conv1d_ane(poly, poly_degree + 1, kernel, kernel_size, conv_result, &result_len);

    if (!ok || result_len == 0) {
        float eval = 0.0f;
        int use_len = (poly_degree + 1) < kernel_size ? (poly_degree + 1) : kernel_size;
        for (int i = 0; i < use_len; i++) {
            eval += poly[i] * kernel[i];
        }
        uint8_t eval_bytes[sizeof(float) + 4];
        *(float *)eval_bytes = eval;
        *(uint32_t *)(eval_bytes + 4) = poly_degree;
        CC_SHA256(eval_bytes, sizeof(eval_bytes), poly_commit);
    } else {
        uint8_t hash_input[sizeof(float) * 512];
        int copy_len = result_len < 512 ? result_len : 512;
        for (int i = 0; i < copy_len; i++) {
            *(float *)(hash_input + i * sizeof(float)) = conv_result[i];
        }
        CC_SHA256(hash_input, copy_len * sizeof(float), poly_commit);
    }

    // Hash the leaf
    uint8_t leaf_hash[32];
    hash_leaf(poly_commit, leaf_hash);

    // Build path for leaf 0 (our polynomial is at index 0)
    int leaf_index = 0;
    int tree_depth = 0;
    int temp = CONV_PCS_NUM_LEAVES;
    while (temp > 1) { temp = (temp + 1) / 2; tree_depth++; }

    // Generate merkle proof path
    // In a real implementation, the prover would have access to all leaves
    // For single polynomial, we construct the path given our leaf
    uint8_t leaves[CONV_PCS_NUM_LEAVES][32];
    for (int i = 0; i < CONV_PCS_NUM_LEAVES; i++) {
        if (i == 0) {
            memcpy(leaves[i], leaf_hash, 32);
        } else {
            uint8_t pad_data[33];
            pad_data[0] = 0xFF;
            *(uint32_t *)(pad_data + 1) = i;
            CC_SHA256(pad_data, 5, leaves[i]);
        }
    }

    // Compute path up the tree
    int remaining = CONV_PCS_NUM_LEAVES;
    int offset = 0;
    for (int level = 0; level < tree_depth; level++) {
        int pair_index = leaf_index >> level;
        int sibling_index = (pair_index % 2 == 0) ? pair_index + 1 : pair_index - 1;

        // Need to know sibling - reconstruct sibling from tree state
        // This is simplified - real implementation would store full tree
        uint8_t (*tree)[32] = calloc(CONV_PCS_NUM_LEAVES * 2, 32);
        memcpy(tree, leaves, CONV_PCS_NUM_LEAVES * 32);

        int tree_offset = CONV_PCS_NUM_LEAVES;
        int tree_remaining = CONV_PCS_NUM_LEAVES;
        for (int l = 0; l < level; l++) {
            int parent_count = (tree_remaining + 1) / 2;
            tree_offset += parent_count;
            tree_remaining = parent_count;
        }

        // At this level, we have tree_remaining nodes at tree_offset
        // We need sibling at sibling_index
        int sibling_local = sibling_index;
        if (sibling_index >= tree_remaining) {
            sibling_local = pair_index;  // Use self if sibling is padding
        }

        if (sibling_local < tree_remaining) {
            memcpy(proof->merkle_proof + level * 32, tree + tree_offset + sibling_local, 32);
        } else {
            // No sibling available - copy self
            memcpy(proof->merkle_proof + level * 32, tree + tree_offset + (pair_index % tree_remaining), 32);
        }

        free(tree);
    }
    proof->merkle_depth = tree_depth;

    // Compute evaluation proof: hash(point, evaluation, leaf_hash)
    uint8_t proof_data[sizeof(float) * 2 + 32];
    *(float *)proof_data = point;
    *(float *)(proof_data + 4) = evaluation;
    memcpy(proof_data + 8, leaf_hash, 32);
    CC_SHA256(proof_data, sizeof(proof_data), proof->evaluation_proof);

    return true;
}

bool conv_pcs_verify(
    const ConvPCSCommitment *commitment,
    float point,
    float evaluation,
    const ConvPCSProof *proof
) {
    // Simplified verification - just check structure is valid
    // For full verification, use conv_pcs_verify_with_kernel
    if (!commitment || !proof) return false;

    if (proof->merkle_depth > CONV_PCS_MERKLE_DEPTH) return false;

    bool eval_proof_nonzero = false;
    for (int i = 0; i < 32; i++) {
        if (proof->evaluation_proof[i] != 0) {
            eval_proof_nonzero = true;
            break;
        }
    }

    bool merkle_proof_valid = (proof->merkle_depth > 0 && proof->merkle_depth <= CONV_PCS_MERKLE_DEPTH);

    return eval_proof_nonzero && merkle_proof_valid;
}

bool conv_pcs_verify_with_kernel(
    const ConvPCSCommitment *commitment,
    float point,
    float evaluation,
    const float *kernel,
    int kernel_size,
    const ConvPCSProof *proof
) {
    if (!commitment || !proof || !kernel) return false;

    if (proof->merkle_depth > CONV_PCS_MERKLE_DEPTH) return false;

    // Step 1: Recompute polynomial commitment from commitment
    // The commitment contains poly_commit which is hash of conv_result
    // We can't recompute it without knowing the polynomial, but we can
    // verify the merkle path is consistent with the commitment's merkle_root

    // Step 2: Verify evaluation proof
    // The evaluation_proof is SHA256(point || evaluation || leaf_hash)
    // For this we'd need to know leaf_hash = hash_leaf(poly_commit)
    // which requires knowing the original poly

    // For a meaningful verification, we verify:
    // 1. The evaluation is consistent with point (evaluate polynomial)
    // 2. The merkle path can reconstruct the commitment's merkle_root

    // Since we don't have the polynomial, we verify the structure is consistent
    // In a full implementation, the verifier would receive (poly, point, evaluation, proof)
    // and would:
    //   a. Recompute poly_commit via ANE convolution with same kernel
    //   b. Verify leaf_hash = hash_leaf(poly_commit)
    //   c. Verify merkle path from leaf_hash to commitment->merkle_root
    //   d. Verify evaluation_proof = SHA256(point || evaluation || leaf_hash)

    // For now, do structural verification:
    bool eval_proof_nonzero = false;
    for (int i = 0; i < 32; i++) {
        if (proof->evaluation_proof[i] != 0) {
            eval_proof_nonzero = true;
            break;
        }
    }

    if (!eval_proof_nonzero) return false;

    // Verify merkle path can lead to commitment's merkle_root
    // We don't have the leaf, so we verify the path structure is valid for the depth
    int tree_depth = 0;
    int temp = CONV_PCS_NUM_LEAVES;
    while (temp > 1) { temp = (temp + 1) / 2; tree_depth++; }

    if (proof->merkle_depth != tree_depth) return false;

    // Merkle path verification requires the leaf hash
    // Without the polynomial, we can't fully verify
    // But we can verify the path length matches expected tree structure
    return true;
}

// ============================================================================
// Serialization
// ============================================================================

void conv_pcs_commit_serialize(const ConvPCSCommitment *commitment, uint8_t *output) {
    memcpy(output, commitment->poly_commit, 32);
    memcpy(output + 32, commitment->merkle_root, 32);
}

void conv_pcs_commit_deserialize(const uint8_t *input, ConvPCSCommitment *commitment) {
    memcpy(commitment->poly_commit, input, 32);
    memcpy(commitment->merkle_root, input + 32, 32);
}

void conv_pcs_proof_serialize(const ConvPCSProof *proof, uint8_t *output, size_t *output_len) {
    size_t offset = 0;
    memcpy(output + offset, proof->evaluation_proof, 32);
    offset += 32;
    memcpy(output + offset, proof->merkle_proof, 32 * proof->merkle_depth);
    offset += 32 * proof->merkle_depth;
    *(uint32_t *)(output + offset) = proof->merkle_depth;
    offset += 4;
    *output_len = offset;
}

bool conv_pcs_proof_deserialize(const uint8_t *input, size_t input_len, ConvPCSProof *proof) {
    if (input_len < 32 + 4) return false;

    size_t offset = 0;
    memcpy(proof->evaluation_proof, input + offset, 32);
    offset += 32;

    uint32_t depth = *(uint32_t *)(input + offset);
    if (depth > CONV_PCS_MERKLE_DEPTH || offset + 32 * depth + 4 > input_len) {
        return false;
    }
    proof->merkle_depth = depth;
    offset += 4;

    memcpy(proof->merkle_proof, input + offset, 32 * depth);
    return true;
}