#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_clamp(op, data_type)                                              \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    data_type *input_ptr = (data_type *)input->values;                         \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *output_ptr = (data_type *)output->values;                       \
    float min, max;                                                            \
    EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_MIN_FLOAT32, min, min);             \
    EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_MAX_FLOAT32, max, max);             \
    cth_tensor_dim_t N = input->meta_info->n_elements;                         \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      if (input_ptr[i] < min) {                                                \
        output_ptr[i] = min;                                                   \
      } else if (input_ptr[i] > max) {                                         \
        output_ptr[i] = max;                                                   \
      } else {                                                                 \
        output_ptr[i] = input_ptr[i];                                          \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * Clamp op: https://pytorch.org/docs/stable/torch.html#torch.clamp
 *
 * Op requirement:
 *    - # of input tensors: 1
 *        - input: the input always at index 0
 *    - # of arguments: 2
 *        - CTH_PARAM_TYPE_MIN_FLOAT32
 *        - CTH_PARAM_TYPE_MAX_FLOAT32
 *    - # of output tensors: 1
 *        - output:  the input always at index 0
 *
 * Note: does not support boradcast
 */
void op_clamp_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  FORCE_OP_PARAM_NUM(op, 2);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_MAX_FLOAT32);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_MIN_FLOAT32);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE data_type = input->meta_info->data_type;

  _cpu_generic_compute(op, _cth_clamp, data_type);
}
