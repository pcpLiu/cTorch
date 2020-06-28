#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_lerp_kernel(op, data_type)                                        \
  do {                                                                         \
    CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *input_3 = array_at(CTorchTensor)(op->in_bound_tensors, 2);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *ptr_1 = (data_type *)input_1->values;                           \
    data_type *ptr_2 = (data_type *)input_2->values;                           \
    data_type *ptr_3 = (data_type *)input_3->values;                           \
    data_type *ptr_output = (data_type *)output->values;                       \
    tensor_size_t N = input_1->meta_info->n_elements;                          \
                                                                               \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      ptr_output[i] = ptr_1[i] + ptr_3[i] * (ptr_2[i] - ptr_1[i]);             \
    }                                                                          \
  } while (0)

/**
 * out = input_1 + input_3 * (input_2 - input_1)
 *
 * Note: unlike PyTorch, no scalar is allowed here
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_lerp_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 3, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);
  // TODO: same type force

  CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_lerp_kernel, input->meta_info->data_type);
}
