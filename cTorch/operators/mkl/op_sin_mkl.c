#include "cTorch/operators/mkl/op_list_mkl.h"
#include "cTorch/operators/mkl/util_mkl.h"
#include <mkl.h>

/**
 * @brief Computes the sin of vector elements
 *
 * @param[CTHOperator] op operator
 *
 * @note MKL only support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 1
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_sin_mkl(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);

  _cth_mkl_vm_function_call_unary(
      input->meta_info->data_type,
      Sin,
      input->values,
      output->values,
      input->meta_info->n_elements);
}
