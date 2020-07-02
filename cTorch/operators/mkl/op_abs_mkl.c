#include "cTorch/operators/mkl/op_list_mkl.h"
#include "cTorch/operators/mkl/util_mkl.h"
#include <mkl.h>

/**
 * @brief Compute the element-wise absolute value of the given input tensor.
 *
 * @param[CTorchOperator] op operator
 *
 * @note MKL onlly support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 1
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_abs_mkl(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE types[2] = {
      CTH_TENSOR_DATA_TYPE_FLOAT_32,
      CTH_TENSOR_DATA_TYPE_FLOAT_64,
  };
  FORCE_TENSOR_TYPES(input, types, 2);

  _cth_mkl_vm_function_call(
      input->meta_info->data_type,
      Abs,
      input->values,
      output->values,
      input->meta_info->n_elements);
}
