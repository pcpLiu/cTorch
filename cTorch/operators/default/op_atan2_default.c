#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_atan2_kernel(op, data_type)                                       \
  do {                                                                         \
    CTorchTensor *input_x = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_y = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *ptr_x = (data_type *)input_x->values;                           \
    data_type *ptr_y = (data_type *)input_y->values;                           \
    data_type *ptr_output = (data_type *)output->values;                       \
    tensor_size_t N = input_x->meta_info->n_elements;                          \
                                                                               \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      ptr_output[i] = atan2(ptr_y[i], ptr_x[i]);                               \
    }                                                                          \
  } while (0)

/**
 * Computation see wiki: https://en.wikipedia.org/wiki/Atan2
 *
 * Inputs & outputs:
 *    - # of input: 2
 *        - 0: x
 *        - 1: y
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_atan2_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_atan2_kernel, input->meta_info->data_type);
}
