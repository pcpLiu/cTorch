#include "cTorch/operators/mkl/op_list_mkl.h"
#include "cTorch/operators/mkl/util_mkl.h"
#include <mkl.h>

/**
 * @brief Each element of the tensor input is multiplied by the corresponding
 * element ofhe Tensor other. The resulting tensor is returned.
 *
 * @note Assume input and output tensor have same datatypes. If not, this op
 * will fail
 *
 * @note MKL only support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 2
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_mul_mkl(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in_a = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *in_b = array_at(CTorchTensor)(op->in_bound_tensors, 1);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  tensor_size_t N = in_a->meta_info->n_elements;
  _cth_mkl_vm_function_call_binary(
      in_a->meta_info->data_type,
      Mul,
      in_a->values,
      in_b->values,
      out->values,
      N);
}
