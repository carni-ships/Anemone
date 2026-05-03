#ifndef ORION_MIL_BUILDER_H
#define ORION_MIL_BUILDER_H

#import <Foundation/Foundation.h>

/// MIL text generator helpers for ANE compilation.
/// Each function generates a block of MIL statements that can be composed.
/// Convention: input tensors are named by caller; output tensor name is returned.

/// Generate the MIL program header (version + buildInfo).
NSString* orion_mil_header(void);

/// Generate a conv-based linear layer: out = conv(W, x) [+ b].
/// Uses 1×1 conv (3× faster than matmul on ANE).
/// @param prefix   Unique name prefix for generated variables
/// @param input    Name of the input tensor (fp16 [1, in_dim, 1, seq])
/// @param in_dim   Input dimension (channels)
/// @param out_dim  Output dimension (channels)
/// @param seq      Sequence length
/// @param weight_path  BLOBFILE path for weight blob (e.g., "@model_path/weights/wq.bin")
/// @param bias_path    BLOBFILE path for bias blob, or NULL for no bias
/// @return MIL statements block. Output tensor is named "{prefix}_out".
NSString* orion_mil_linear(const char* prefix, const char* input,
                           int in_dim, int out_dim, int seq,
                           const char* weight_path, const char* bias_path);

/// Generate LayerNorm: out = gamma * (x - mean) / sqrt(var + eps) + beta.
/// @param prefix     Unique name prefix
/// @param input      Input tensor name (fp16 [1, dim, 1, seq])
/// @param dim        Channel dimension
/// @param seq        Sequence length
/// @param gamma_path BLOBFILE path for gamma weights
/// @param beta_path  BLOBFILE path for beta weights
/// @param eps        Epsilon for numerical stability
/// @return MIL statements. Output: "{prefix}_out".
NSString* orion_mil_layernorm(const char* prefix, const char* input,
                              int dim, int seq,
                              const char* gamma_path, const char* beta_path, float eps);

/// Generate RMSNorm (Llama-style): out = w * x * rsqrt(mean(x^2) + eps).
/// @param prefix     Unique name prefix
/// @param input      Input tensor name (fp16 [1, dim, 1, seq])
/// @param dim        Channel dimension
/// @param seq        Sequence length
/// @param weight_path BLOBFILE path for RMSNorm weight
/// @param eps        Epsilon
/// @return MIL statements. Output: "{prefix}_out".
NSString* orion_mil_rmsnorm(const char* prefix, const char* input,
                            int dim, int seq,
                            const char* weight_path, float eps);

/// Generate GELU activation: out = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
/// @param prefix  Unique name prefix
/// @param input   Input tensor name
/// @param dim     Channel dimension
/// @param seq     Sequence length
/// @return MIL statements. Output: "{prefix}_out".
NSString* orion_mil_gelu(const char* prefix, const char* input, int dim, int seq);

/// Generate SiLU activation: out = x * sigmoid(x).
/// @param prefix  Unique name prefix
/// @param input   Input tensor name
/// @param dim     Channel dimension
/// @param seq     Sequence length
/// @return MIL statements. Output: "{prefix}_out".
NSString* orion_mil_silu(const char* prefix, const char* input, int dim, int seq);

/// Generate causal attention with explicit decomposition (NOT using SDPA).
/// Q@K^T / sqrt(d) → apply causal mask → softmax → @V.
/// @param prefix   Unique name prefix
/// @param q_input  Name of Q tensor (fp16 [1, d_model, 1, seq])
/// @param k_input  Name of K tensor
/// @param v_input  Name of V tensor
/// @param n_head   Number of attention heads
/// @param head_dim Per-head dimension
/// @param seq      Sequence length
/// @param mask_path BLOBFILE path for causal mask blob
/// @return MIL statements. Output: "{prefix}_out".
NSString* orion_mil_causal_attention(const char* prefix,
                                     const char* q_input, const char* k_input, const char* v_input,
                                     int n_head, int head_dim, int seq,
                                     const char* mask_path);

/// Wrap MIL body into a complete program with function signature.
/// @param body       Generated MIL statements (concatenated blocks)
/// @param inputs     Array of input declarations, e.g., @"tensor<fp32, [1,256,1,64]> x"
/// @param output_var Name of the output variable in body
/// @return Complete MIL program text.
NSString* orion_mil_program(NSString* body, NSArray<NSString*>* inputs, NSString* output_var);

/// Wrap MIL body with multiple outputs.
/// @param body        Generated MIL statements
/// @param inputs      Array of input declarations
/// @param output_vars Array of output variable names
/// @return Complete MIL program text.
NSString* orion_mil_program_multi(NSString* body, NSArray<NSString*>* inputs, NSArray<NSString*>* output_vars);

/// Generate causal mask BLOBFILE data (fp16 [seq, seq], 0 for j<=i, -1e4 for j>i).
/// Returns NSData containing a valid BLOBFILE (128-byte header + fp16 data).
NSData* orion_make_causal_mask_blob(int seq_len);

/// Return MIL-style blob path for causal mask: "@model_path/masks/causal_{seq_len}.bin"
NSString* orion_causal_mask_path(int seq_len);

#pragma mark - T024: Batch Polynomial Evaluation (Horner)

/// Generate Horner evaluation for N polynomials at the same point.
/// Uses chain of multiply-accumulate: P(x) = c_0 + x*(c_1 + x*(c_2 + ...))
/// @param prefix   Unique name prefix
/// @param n_polys  Number of polynomials to evaluate
/// @param degree   Degree of each polynomial (number of coefficients - 1)
/// @param seq      Sequence length (batch dimension)
/// @param coeff_path BLOBFILE path for coefficients [n_polys, (degree+1)*n_polys]
/// @return MIL statements. Output: "{prefix}_out" [1, n_polys, 1, seq]
NSString* orion_mil_poly_eval_horner(const char* prefix, int n_polys, int degree,
                                     int seq, const char* coeff_path);

#pragma mark - T025: Inner Product

/// Create weight blob for inner product: diagonal matrix with b values
/// @param b    Input vector b (will become diagonal)
/// @param n    Length of vectors
/// @return NSData with 128-byte header + fp16 data
NSData* orion_make_inner_product_blob(const float *b, int n);

/// Generate inner product <a,b> = sum(a_i * b_i) using conv1x1.
/// Input a [1, n, 1, seq], b is baked into diagonal weight matrix.
/// @param prefix  Unique name prefix
/// @param n       Vector length
/// @param seq     Sequence length
/// @param a_input Name of input tensor a
/// @param b_path  BLOBFILE path for diagonal weight blob
/// @return MIL statements. Output: "{prefix}_out" [1, 1, 1, seq]
NSString* orion_mil_inner_product(const char* prefix, int n, int seq,
                                  const char* a_input, const char* b_path);

#pragma mark - T026: Matrix-Matrix Multiplication (Batched MatVec)

/// Generate MatMat multiplication: C = A * B where A is k×l and B is l×m
/// @param prefix  Unique name prefix
/// @param k       A matrix rows (also C rows)
/// @param l       A matrix cols (also B rows)
/// @param m       B matrix cols (also C cols)
/// @param seq     Sequence length
/// @param a_path  BLOBFILE path for A weights [k, l, 1, 1]
/// @param b_input Name of input tensor B [1, l, 1, m]
/// @return MIL statements. Output: "{prefix}_out" [1, k, 1, m]
NSString* orion_mil_matmat(const char* prefix, int k, int l, int m, int seq,
                           const char* a_path, const char* b_input);

#pragma mark - T027: NTT Butterfly (small N only, N <= 16)

/// Generate single NTT butterfly stage: (a,b) -> (a+b*w, a-b*w)
/// For N <= 16, twiddle factors can be baked into weights.
/// @param prefix    Unique name prefix
/// @param n         Transform size (must be <= 16)
/// @param seq       Sequence length
/// @param tw_path   BLOBFILE path for twiddle weights (n×n diagonal matrix)
/// @return MIL statements. Output: "{prefix}_out" [1, n, 1, seq]
NSString* orion_mil_ntt_butterfly(const char* prefix, int n, int seq,
                                   const char* tw_path);

/// Pure butterfly MIL program (no twiddles)
/// Computes add/sub for NTT butterfly stages on ANE
/// @param prefix    Unique name prefix
/// @param n         Transform size
/// @param weight_path BLOBFILE path for butterfly weights
/// @return MIL statements. Output: "{prefix}_y" [1, n, 1, 1]
NSString* orion_mil_ntt_pure_butterfly(const char* prefix, int n,
                                        const char* weight_path);

#endif // ORION_MIL_BUILDER_H
