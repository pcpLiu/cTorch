#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"
#include <tgmath.h>

#define _cth_pow_kernel(a, b) (pow(a, b))

/**
 * @brief Takes the power of each element in input with exponent and returns a
 * tensor with the result.
 *
 * @par In implementation, all input & output were converted into float doing
 * the pow(). Then converted back.
 *
 * @param op operator
 *
 * @note Two inputs must have same dimensions.
 *
 * Inputs & outputs:
 * - # of input: 2
 *  - 0: base tensor
 *  - 1: exponent tensor
 * - # of outputs: 1
 */
void op_pow_cpu(CTHOperator *op) {
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
      _cth_pow_kernel);
}
