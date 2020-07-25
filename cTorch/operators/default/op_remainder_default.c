#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"
#include <tgmath.h>

/**
 * @brief Computes the element-wise remainder of division.
 *
 * @par The divisor and dividend may contain both for integer and floating point
 * numbers. The remainder has the same sign as the divisor.
 *
 * Inputs & outputs:
 *  - # of input: 2
 *    - 0: dividend tensor
 *    - 1: the divisor
 *  - # of output: 1
 */
void op_remainder_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *in_a = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *in_b = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  cth_tensor_dim_t N = in_a->meta_info->n_elements;
  _cpu_1d_map_elewise_binary(
      in_a->values,
      in_b->values,
      out->values,
      in_a->meta_info->data_type,
      N,
      remainder);
}
