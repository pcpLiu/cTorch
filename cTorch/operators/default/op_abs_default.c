#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * @brief Compute the element-wise absolute value of the given input tensor.
 *
 * @param[CTHOperator] op operator
 *
 * @note Do not support boolean operator
 *
 * Inputs & outputs:
 *   - # of input: 1
 *   - # of output: 1
 */
void op_abs_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  cth_tensor_dim_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary(
      in->values, out->values, in->meta_info->data_type, N, fabs);
}
