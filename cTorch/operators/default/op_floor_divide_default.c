#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_floor_divide(op, data_type)                                       \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *input_1_ptr = (data_type *)input_1->values;                     \
    data_type *input_2_ptr = (data_type *)input_2->values;                     \
    data_type *output_ptr = (data_type *)output->values;                       \
    cth_tensor_dim_t N = input_1->meta_info->n_elements;                       \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      output_ptr[i] = (data_type)floor(1.0 * input_1_ptr[i] / input_2_ptr[i]); \
    }                                                                          \
  } while (0);

/**
 * Computes exp(x) -1
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_floor_divide_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE data_type = in->meta_info->data_type;
  _cpu_generic_compute(op, _cth_floor_divide, data_type);
}
