#include "cTorch/operators/mkl/op_list_mkl.h"
#include "cTorch/operators/mkl/util_mkl.h"
#include <mkl.h>

/**
 * @brief Computes the inverse sine of vector elements
 *
 * @param[CTorchOperator] op operator
 *
 * @note MKL only support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 1
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_asin_mkl(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);

  _cth_mkl_vm_function_call_unary(
      input->meta_info->data_type,
      Asin,
      input->values,
      output->values,
      input->meta_info->n_elements);
}
