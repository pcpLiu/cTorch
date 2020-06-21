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
void op_add_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int64_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_binary(
      in->values, out->values, in->meta_info->data_type, N, _cth_add_kernel);
}