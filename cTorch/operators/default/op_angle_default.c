
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_angle_kernel(input_ptr, output_ptr, N, data_type)                 \
  do {                                                                         \
    data_type *input_t = (data_type *)input_ptr;                               \
    data_type *output_t = (data_type *)output_ptr;                             \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = (data_type)((float)input_t[i] / CTH_PI);                   \
    }                                                                          \
  } while (0)

/**
 * Computes the element-wise angle (in radians) of the given input tensor.
 * Computation: input / Pi
 *
 * Note:
 * During computation, input values will be converted to float first and then
 * final result will be stored into output tensor as its datatype.
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.

 */
void op_angle_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);
  // TODO: same type force

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int64_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary_generic(
      in->values, out->values, in->meta_info->data_type, N, _cth_angle_kernel);
}
