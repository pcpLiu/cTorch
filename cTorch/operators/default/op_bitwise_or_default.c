#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_bitwise_or(op, data_type)                                         \
  do {                                                                         \
    CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    data_type *out_ptr = (data_type *)output->values;                          \
    tensor_size_t N = input_1->meta_info->n_elements;                          \
                                                                               \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      out_ptr[i] = in_ptr_1[i] | in_ptr_2[i];                                  \
    }                                                                          \
  } while (0)

/**
 * Computes the bitwise OR of the given two input tensor.
 * The input tensor must be of integral or Boolean types. For bool tensors, it
 * computes the logical OR.
 *
 * # of input: 2
 * # of output: 1
 */
void op_bitwise_or_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  CTorchTensor *in_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *in_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  tensor_size_t N = in_1->meta_info->n_elements;
  CTH_TENSOR_DATA_TYPE data_type = in_1->meta_info->data_type;

  CTH_TENSOR_DATA_TYPE types[5] = {
      CTH_TENSOR_DATA_TYPE_BOOL,
      CTH_TENSOR_DATA_TYPE_INT_16,
      CTH_TENSOR_DATA_TYPE_INT_32,
      CTH_TENSOR_DATA_TYPE_INT_64,
      CTH_TENSOR_DATA_TYPE_UINT_8,
  };
  FORCE_TENSOR_TYPES(in_1, types, 5);
  FORCE_TENSOR_TYPES(in_2, types, 5);
  FORCE_TENSOR_TYPES(out, types, 5);

  _cpu_bit_compute(op, _cth_bitwise_or, data_type);
}
