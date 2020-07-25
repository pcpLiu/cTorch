#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_logical_and(op, data_type)                                        \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    bool *out_ptr = (bool *)output->values;                                    \
    cth_tensor_dim_t N = input_1->meta_info->n_elements;                       \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      out_ptr[i] = (in_ptr_1[i] == 0 ? 0 : 1) == (in_ptr_2[i] == 0 ? 0 : 1);   \
    }                                                                          \
  } while (0)

/**
 * Computes the element-wise logical AND of the given input tensors. Zeros are
 * treated as False and nonzeros are treated as True.
 *
 * Note: Both inputs should have same data types.
 *
 * # of input: 2
 * # of output: 1
 *    - Must be bool type
 */
void op_logical_and_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);

  CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE types[1] = {CTH_TENSOR_DATA_TYPE_BOOL};
  CTH_FORCE_TENSOR_TYPES(output, types, 1);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_logical_and, input->meta_info->data_type);
}
