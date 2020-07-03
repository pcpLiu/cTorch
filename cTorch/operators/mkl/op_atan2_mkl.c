#include "cTorch/operators/mkl/op_list_mkl.h"
#include "cTorch/operators/mkl/util_mkl.h"
#include <mkl.h>

/**
 * @brief Computes the four-quadrant inverse tangent of ratios of the elements
 * of two vectors
 *
 * @param[CTorchOperator] op operator
 *
 * @note MKL only support float & double type
 *
 * @note The second input, is the x-coordinate, while the first parameter is the
 * y-coordinate.
 *
 * Inputs & outputs:
 *    - # of input: 2
 *        - 0: x
 *        - 1: y
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_atan2_mkl(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in_a = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *in_b = array_at(CTorchTensor)(op->in_bound_tensors, 1);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int64_t N = in_a->meta_info->n_elements;
  _cth_mkl_vm_function_call_binary(
      in_a->meta_info->data_type,
      Atan2,
      in_b->values,
      in_a->values,
      out->values,
      N);
}
