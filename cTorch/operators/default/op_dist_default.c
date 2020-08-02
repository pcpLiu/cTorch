#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"
#include <tgmath.h>

#define _cth_dist(op, data_type, input_type_enum, output_type_enum)            \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    data_type *input_ptr_1 = (data_type *)input_1->values;                     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    data_type *input_ptr_2 = (data_type *)input_2->values;                     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *output_ptr = (data_type *)output->values;                       \
    float p;                                                                   \
    EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_P_FLOAT32, p, p);                   \
                                                                               \
    cth_tensor_dim_t N = input->meta_info->n_elements;                         \
    data_type dist = 0;                                                        \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      dist += pow(fabs((float)input_ptr_1[i] - (float)input_ptr_2[i]), p);     \
    }                                                                          \
    CTH_LOG(CTH_LOG_ERR, "pow(dist, 1.0 / p): %lf", pow(dist, 1.0 / p));       \
    if (output_type_enum == CTH_TENSOR_DATA_TYPE_INT_16 ||                     \
        output_type_enum == CTH_TENSOR_DATA_TYPE_INT_32 ||                     \
        output_type_enum == CTH_TENSOR_DATA_TYPE_INT_64 ||                     \
        output_type_enum == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      output_ptr[0] = round(pow(dist, 1.0 / p));                               \
    } else {                                                                   \
      output_ptr[0] = (data_type)pow(dist, 1.0 / p);                           \
    }                                                                          \
  } while (0);

/**
 * out = p-norm dist (input_1 - input_2)
 * ref: https://pytorch.org/docs/stable/torch.html#torch.addcmul
 *
 * Op requirement:
 *    - # of input tensors: 2
 *        - input_1: the input always at index 0
 *        - input_2: the input always at index 1
 *    - # of arguments: 1
 *        - CTH_PARAM_TYPE_P_FLOAT32
 *    - # of output tensors: 1
 *        - The output tensor should just has 1 element
 *
 * Note: does not support boradcast
 */
void op_dist_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  FORCE_OP_PARAM_NUM(op, 1);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_P_FLOAT32);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);

  _cpu_generic_compute2(
      op, _cth_dist, input->meta_info->data_type, input->meta_info->data_type);
}
