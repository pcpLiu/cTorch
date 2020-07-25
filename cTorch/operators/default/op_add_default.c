#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_add_kernel(a, b) (a + b)

/**
 * Add two input tensors and store result in output tensor. No broadcast
 * support.
 *
 * Assume input and output tensor have same datatypes. If not, this op will
 * fail.
 */
void op_add_cpu(CTHOperator *op) {
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
      _cth_add_kernel);
}
