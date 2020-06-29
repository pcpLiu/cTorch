#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_logical_or(op, data_type)                                         \
  do {                                                                         \
    CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    bool *out_ptr = (bool *)output->values;                                    \
    tensor_size_t N = input_1->meta_info->n_elements;                          \
                                                                               \
    uint16_t x, y;                                                             \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      x = (in_ptr_1[i] == 0 ? 0 : 1);                                          \
      y = (in_ptr_2[i] == 0 ? 0 : 1);                                          \
      out_ptr[i] = (x == 1 || y == 1 ? true : false);                          \
    }                                                                          \
  } while (0)

/**
 * Computes the element-wise logical OR of the given input tensors. Zeros are
 * treated as False and nonzeros are treated as True.
 *
 * Note: Both inputs should have same data types.
 *
 * # of input: 2
 * # of output: 1
 *    - Must be bool type
 */
void op_logical_or_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);

  CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE types[1] = {CTH_TENSOR_DATA_TYPE_BOOL};
  FORCE_TENSOR_TYPES(output, types, 1);

  CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_logical_or, input->meta_info->data_type);
}
