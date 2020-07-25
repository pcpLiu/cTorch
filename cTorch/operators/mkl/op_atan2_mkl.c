#include "cTorch/operators/mkl/op_list_mkl.h"
#include "cTorch/operators/mkl/util_mkl.h"
#include <mkl.h>

/**
 * @brief Computes the four-quadrant inverse tangent of ratios of the elements
 * of two vectors
 *
 * @param[CTHOperator] op operator
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
void op_atan2_mkl(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *in_a = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *in_b = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  cth_tensor_dim_t N = in_a->meta_info->n_elements;
  _cth_mkl_vm_function_call_binary(
      in_a->meta_info->data_type,
      Atan2,
      in_b->values,
      in_a->values,
      out->values,
      N);
}
