#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define cth_neg(x) (-x)

/**
 * @brief Compute the negative of the elements of input.
 *
 * @param[CTorchOperator] op operator
 *
 * @note Do not support boolean operator
 *
 * Inputs & outputs:
 *   - # of input: 1
 *   - # of output: 1
 */
void op_neg_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int64_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary(
      in->values, out->values, in->meta_info->data_type, N, cth_neg);
}