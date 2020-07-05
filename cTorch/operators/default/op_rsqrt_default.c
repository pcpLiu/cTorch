#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"
#include <tgmath.h>

#define _cth_rsqrt_kernel(x) (1 / sqrt(x))

/**
 * @brief Returns a new tensor with the reciprocal of the square-root of each of
 * the elements of input.
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_rsqrt_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);
  // TODO: same type force

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  tensor_size_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary(
      in->values, out->values, in->meta_info->data_type, N, _cth_rsqrt_kernel);
}
