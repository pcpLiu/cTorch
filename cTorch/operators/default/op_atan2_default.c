#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_atan2_kernel(op, data_type)                                       \
  do {                                                                         \
    CTorchTensor *input_x = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    data_type *ptr_x = (data_type *)input_x->values;                           \
    CTorchTensor *input_y = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    data_type *ptr_y = (data_type *)input_y->values;                           \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *ptr_output = (data_type *)output->values;                       \
    tensor_size_t N = tensor_1->meta_info->n_elements;                         \
                                                                               \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      data_type x = ptr_x[i];                                                  \
      data_type y = ptr_y[i];                                                  \
      if (x > 0.0) {                                                           \
        ptr_output[i] = arctan;                                                \
      } else if (y > 0.0) {                                                    \
      } else if (y < 0.0) {                                                    \
      } else if (x < 0.0) {                                                    \
      } else {                                                                 \
        FAIL_EXIT(CTH_LOG_ERR, "op_atan2_cpu meets undefined situation");      \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * Computation see wiki: https://en.wikipedia.org/wiki/Atan2
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_atan2_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_atan2_kernel, input->meta_info->data_type);
}
