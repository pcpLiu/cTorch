#define COMMA ,
#define SEMI_COL ;

/*
  Number of enabled operators
*/
#define ENABLED_OP_NUM 297

// clang-format off

/*******************************************************************************
 *
 * List of enabled operators
 */
#define FOREACH_OP_ID(wrap) \
  wrap(abs) \
  wrap(acos) \
  wrap(AdaptiveAvgPool1d) \
  wrap(AdaptiveAvgPool2d) \
  wrap(AdaptiveAvgPool3d) \
  wrap(AdaptiveMaxPool1d) \
  wrap(AdaptiveMaxPool2d) \
  wrap(AdaptiveMaxPool3d) \
  wrap(add) \
  wrap(addbmm) \
  wrap(addcdiv) \
  wrap(addcmul) \
  wrap(addmm) \
  wrap(addmv) \
  wrap(addr) \
  wrap(allclose) \
  wrap(angle) \
  wrap(argmax) \
  wrap(argmin) \
  wrap(argsort) \
  wrap(asin) \
  wrap(atan) \
  wrap(atan2) \
  wrap(AvgPool1d) \
  wrap(AvgPool2d) \
  wrap(AvgPool3d) \
  wrap(baddbmm) \
  wrap(bartlett_window) \
  wrap(BatchNorm1d) \
  wrap(BatchNorm2d) \
  wrap(BatchNorm3d) \
  wrap(BCELoss) \
  wrap(BCEWithLogitsLoss) \
  wrap(bernoulli) \
  wrap(Bilinear) \
  wrap(bincount) \
  wrap(bitwise_not) \
  wrap(bitwise_and) \
  wrap(bitwise_or) \
  wrap(bitwise_xor) \
  wrap(blackman_window) \
  wrap(bmm) \
  wrap(broadcast_tensors) \
  wrap(cartesian_prod) \
  wrap(cat) \
  wrap(cdist) \
  wrap(ceil) \
  wrap(CELU) \
  wrap(chain_matmul) \
  wrap(cholesky_inverse) \
  wrap(cholesky_solve) \
  wrap(cholesky) \
  wrap(chunk) \
  wrap(clamp) \
  wrap(combinations) \
  wrap(ConstantPad1d) \
  wrap(ConstantPad2d) \
  wrap(ConstantPad3d) \
  wrap(Conv1d) \
  wrap(Conv2d) \
  wrap(Conv3d) \
  wrap(ConvTranspose1d) \
  wrap(ConvTranspose2d) \
  wrap(ConvTranspose3d) \
  wrap(cos) \
  wrap(cosh) \
  wrap(CosineEmbeddingLoss) \
  wrap(CosineSimilarity) \
  wrap(cross) \
  wrap(CrossEntropyLoss) \
  wrap(CTCLoss) \
  wrap(cumprod) \
  wrap(cumsum) \
  wrap(default_generator) \
  wrap(det) \
  wrap(diag_embed) \
  wrap(diag) \
  wrap(diagflat) \
  wrap(diagonal) \
  wrap(digamma) \
  wrap(dist) \
  wrap(div) \
  wrap(dot) \
  wrap(eig) \
  wrap(einsum) \
  wrap(ELU) \
  wrap(Embedding) \
  wrap(EmbeddingBag) \
  wrap(eq) \
  wrap(equal) \
  wrap(erf) \
  wrap(erfc) \
  wrap(erfinv) \
  wrap(exp) \
  wrap(expm1) \
  wrap(fft) \
  wrap(flatten) \
  wrap(flip) \
  wrap(floor) \
  wrap(fmod) \
  wrap(Fold) \
  wrap(frac) \
  wrap(FractionalMaxPool2d) \
  wrap(gather) \
  wrap(ge) \
  wrap(geqrf) \
  wrap(ger) \
  wrap(get_rng_state) \
  wrap(GroupNorm) \
  wrap(GRU) \
  wrap(GRUCell) \
  wrap(gt) \
  wrap(hamming_window) \
  wrap(hann_window) \
  wrap(Hardshrink) \
  wrap(Hardtanh) \
  wrap(HingeEmbeddingLoss) \
  wrap(histc) \
  wrap(Identity) \
  wrap(ifft) \
  wrap(index_select) \
  wrap(initial_seed) \
  wrap(InstanceNorm1d) \
  wrap(InstanceNorm2d) \
  wrap(InstanceNorm3d) \
  wrap(inverse) \
  wrap(irfft) \
  wrap(isfinite) \
  wrap(isinf) \
  wrap(isnan) \
  wrap(KLDivLoss) \
  wrap(kthvalue) \
  wrap(L1Loss) \
  wrap(LayerNorm) \
  wrap(le) \
  wrap(LeakyReLU) \
  wrap(lerp) \
  wrap(lgamma) \
  wrap(Linear) \
  wrap(LocalResponseNorm) \
  wrap(log) \
  wrap(log10) \
  wrap(log1p) \
  wrap(log2) \
  wrap(logdet) \
  wrap(logical_not) \
  wrap(logical_xor) \
  wrap(LogSigmoid) \
  wrap(LogSoftmax) \
  wrap(logsumexp) \
  wrap(LPPool1d) \
  wrap(LPPool2d) \
  wrap(LSTM) \
  wrap(LSTMCell) \
  wrap(lstsq) \
  wrap(lt) \
  wrap(lu_solve) \
  wrap(lu_unpack) \
  wrap(lu) \
  wrap(manual_seed) \
  wrap(MarginRankingLoss) \
  wrap(masked_select) \
  wrap(matmul) \
  wrap(matrix_power) \
  wrap(matrix_rank) \
  wrap(max) \
  wrap(MaxPool1d) \
  wrap(MaxPool2d) \
  wrap(MaxPool3d) \
  wrap(MaxUnpool1d) \
  wrap(MaxUnpool2d) \
  wrap(MaxUnpool3d) \
  wrap(mean) \
  wrap(median) \
  wrap(meshgrid) \
  wrap(min) \
  wrap(mm) \
  wrap(mode) \
  wrap(MSELoss) \
  wrap(mul) \
  wrap(MultiheadAttention) \
  wrap(MultiLabelMarginLoss) \
  wrap(MultiLabelSoftMarginLoss) \
  wrap(MultiMarginLoss) \
  wrap(multinomial) \
  wrap(mv) \
  wrap(mvlgamma) \
  wrap(narrow) \
  wrap(ne) \
  wrap(neg) \
  wrap(NLLLoss) \
  wrap(nonzero) \
  wrap(norm) \
  wrap(normal) \
  wrap(orgqr) \
  wrap(ormqr) \
  wrap(PairwiseDistance) \
  wrap(pinverse) \
  wrap(PixelShuffle) \
  wrap(PoissonNLLLoss) \
  wrap(polygamma) \
  wrap(pow) \
  wrap(PReLU) \
  wrap(prod) \
  wrap(qr) \
  wrap(rand_like) \
  wrap(rand) \
  wrap(randint_like) \
  wrap(randint) \
  wrap(randn_like) \
  wrap(randn) \
  wrap(randperm) \
  wrap(reciprocal) \
  wrap(ReflectionPad1d) \
  wrap(ReflectionPad2d) \
  wrap(ReLU) \
  wrap(ReLU6) \
  wrap(remainder) \
  wrap(renorm) \
  wrap(repeat_interleave) \
  wrap(ReplicationPad1d) \
  wrap(ReplicationPad2d) \
  wrap(ReplicationPad3d) \
  wrap(reshape) \
  wrap(rfft) \
  wrap(RNN) \
  wrap(RNNCell) \
  wrap(roll) \
  wrap(rot90) \
  wrap(round) \
  wrap(RReLU) \
  wrap(rsqrt) \
  wrap(seed) \
  wrap(SELU) \
  wrap(set_rng_state) \
  wrap(Sigmoid) \
  wrap(sigmoid) \
  wrap(sign) \
  wrap(sin) \
  wrap(sinh) \
  wrap(slogdet) \
  wrap(SmoothL1Loss) \
  wrap(SoftMarginLoss) \
  wrap(Softmax) \
  wrap(Softmax2d) \
  wrap(Softmin) \
  wrap(Softplus) \
  wrap(Softshrink) \
  wrap(Softsign) \
  wrap(solve) \
  wrap(sort) \
  wrap(split) \
  wrap(sqrt) \
  wrap(squeeze) \
  wrap(stack) \
  wrap(std_mean) \
  wrap(std) \
  wrap(stft) \
  wrap(sum) \
  wrap(svd) \
  wrap(symeig) \
  wrap(SyncBatchNorm) \
  wrap(t) \
  wrap(take) \
  wrap(tan) \
  wrap(Tanh) \
  wrap(tanh) \
  wrap(Tanhshrink) \
  wrap(tensordot) \
  wrap(Threshold) \
  wrap(topk) \
  wrap(trace) \
  wrap(Transformer) \
  wrap(TransformerDecoderLayer) \
  wrap(TransformerEncoder) \
  wrap(TransformerEncoderLayer) \
  wrap(transpose) \
  wrap(trapz) \
  wrap(triangular_solve) \
  wrap(tril_indices) \
  wrap(tril) \
  wrap(TripletMarginLoss) \
  wrap(triu_indices) \
  wrap(triu) \
  wrap(trunc) \
  wrap(unbind) \
  wrap(Unfold) \
  wrap(unique_consecutive) \
  wrap(unique) \
  wrap(unsqueeze) \
  wrap(Upsample) \
  wrap(UpsamplingBilinear2d) \
  wrap(UpsamplingNearest2d) \
  wrap(var_mean) \
  wrap(var) \
  wrap(where) \
  wrap(ZeroPad2d)
