#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_mul_kernel(a, b) (a * b)

/**
 * @brief Each element of the tensor input is multiplied by the corresponding
 * element ofhe Tensor other. The resulting tensor is returned.
 *
 * @param op operator
 *
 * @note Assume input and output tensor have same datatypes. If not, this op
 * will fail.
 */
void op_mul_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);
  // TODO: check dat type match

  CTorchTensor *in_a = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *in_b = array_at(CTorchTensor)(op->in_bound_tensors, 1);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  tensor_size_t N = in_a->meta_info->n_elements;
  _cpu_1d_map_elewise_binary(
      in_a->values,
      in_b->values,
      out->values,
      in_a->meta_info->data_type,
      N,
      _cth_mul_kernel);
}
