// #include "cTorch/operators/default/op_list.h"
// #include "cTorch/operators/default/util.h"
// #include "cTorch/operators/op_util.h"

// #define _cth_conv1d(op, data_type)                                             \
//   do {                                                                         \
//     CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
//     CTHTensor *weight = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);      \
//     CTHTensor *bias = cth_array_at(CTHTensor)(op->in_bound_tensors, 2);        \
//     CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
//                                                                                \
//     cth_channel_t *in_channels, *out_channels;                                 \
//     cth_kernel_t *kernel_size;                                                 \
//     cth_groups_t *groups;                                                      \
//     cth_stride_t *stride;                                                      \
//     cth_pad_t *padding_d2;                                                     \
//     cth_dilation_t *dilation;                                                  \
//     CTH_PADDING_MODE *padding_mode;                                            \
//                                                                                \
//     EXTRACT_PARAM_VALUE(                                                       \
//         op, CTH_PARAM_TYPE_IN_CHANNELS, in_channels, in_channels);             \
//     EXTRACT_PARAM_VALUE(                                                       \
//         op, CTH_PARAM_TYPE_OUT_CHANNELS, out_channels, out_channels);          \
//     EXTRACT_PARAM_VALUE(                                                       \
//         op, CTH_PARAM_TYPE_KERNEL_SIZE, kernel_size, kernel_size);             \
//     EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_GROUPS, groups, groups);            \
//     EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_STRIDE, stride, stride);            \
//     EXTRACT_PARAM_VALUE(                                                       \
//         op, CTH_PARAM_TYPE_PADDING_D2, padding_d2, padding_d2);                \
//     EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_DILATION, dilation, dilation);      \
//     EXTRACT_PARAM_VALUE(                                                       \
//         op, CTH_PARAM_TYPE_PADDING_MODE, padding_mode, padding_mode);          \
//                                                                                \
//     data_type *out_ptr = (data_type *)output->values;                          \
//     data_type *in_ptr = (data_type *)input->values;                            \
//     data_type *weight_ptr = (data_type *)weight->values;                       \
//     data_type *bias_ptr = (data_type *)bias->values;                           \
//     cth_tensor_dim_t input_channels = input->meta_info->dims[1];               \
//     cth_tensor_dim_t input_x_size = input->meta_info->dims[2];                 \
//     cth_tensor_dim_t input_chennels_per_group =                                \
//         (cth_tensor_dim_t)(input_channels / *groups);                          \
//     cth_tensor_dim_t kernel_nums = weight->meta_info->dims[0];                 \
//     cth_tensor_dim_t kernel_nums_per_group =                                   \
//         (cth_tensor_dim_t)(kernel_nums / *groups);                             \
//     cth_channel_t kernel_channels = weight->meta_info->dims[1];                \
//     cth_tensor_dim_t kernel_size = weight->meta_info->dims[2];                 \
//     cth_tensor_dim_t output_feature_dim = output->meta_info->dims[2];          \
//     cth_tensor_dim_t batches = output->meta_info->dims[0];                     \
//     cth_tensor_dim_t padding_left = PTR_VAL(padding_d2[0]);                    \
//     cth_tensor_dim_t padding_right = PTR_VAL(padding_d2[1]);                   \
//     data_type out_val = 0;                                                     \
//     for (cth_tensor_dim_t bacth_i = 0; batch_i < batches; bacth_i++) {         \
//       for (cth_groups_t group_i = 0; group_i < PTR_VAL(groups); group_i++) {   \
//         for (cth_tensor_dim_t kernel_i = group_i * kernel_nums_per_group;      \
//              kernel_i < (group_i + 1) * kernel_nums_per_group;                 \
//              kernel_i++) {                                                     \
//           for (cth_tensor_dim_t output_x = 0; output_x < output_feature_dim;   \
//                output_x++) {                                                   \
//             data_type out_val = 0;                                             \
//             for (cth_channel_t input_z = group_i * input_chennels_per_group;   \
//                  input_z < (group_i + 1) * group_i;                            \
//                  input_z++) {                                                  \
//               cth_channel_t kernel_z = input_z % input_chennels_per_group;     \
//               for (cth_tensor_dim_t kernel_x = 0; kernel_x < kernel_size;      \
//                    kernel_x++) {                                               \
//                 data_type input_val = 0;                                       \
//                 _cth_get_input_value_1d_conv(                                  \
//                     bacth_i,                                                   \
//                     output_x,                                                  \
//                     input_z,                                                   \
//                     kernel_i,                                                  \
//                     kernel_x,                                                  \
//                     PTR_VAL(padding_mode),                                     \
//                     padding_left,                                              \
//                     padding_right,                                             \
//                     kernel_size,                                               \
//                     PTR_VAL(dilation),                                         \
//                     PTR_VAL(stride),                                           \
//                     in_ptr,                                                    \
//                     input_x_size,                                              \
//                     input_val);                                                \
//                 data_type kernel_val = weight_ptr[];                           \
//                 out_val += kernel_val * input_val;                             \
//               }                                                                \
//             }                                                                  \
//             out_ptr[] = out_val + bias_ptr[];                                  \
//           }                                                                    \
//         }                                                                      \
//       }                                                                        \
//     }                                                                          \
//   } while (0);

// /**
//  * @brief Applies a 1D convolution over an input signal composed of several
//  * input planes. A naive implementation.
//  *
//  * @param[CTHOperator] op operator
//  *
//  * @note Do not support boolean operator
//  *
//  * @par Inputs & outputs:
//  *   - # of input: 3
//  *    - 0: input tensor [batch, in_channels, in_feature_dim]
//  *    - 1: weight tensor [out_channels, in_channels/groups, kernel_size]
//  *    - 2: bias tensor [out_channels]
//  *   - # of output: 1
//  *    - 0: output tensor [batch, out_channels, out_feature_dim]
//  */
// void op_Conv1d_cpu(CTHOperator *op) {
//   FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 3, 1);
//   FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_IN_CHANNELS);
//   FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_OUT_CHANNELS);
//   FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_KERNEL_SIZE_D2);

//   CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
//   CTH_TENSOR_DATA_TYPE data_type = input->meta_info->data_type;
//   FORCE_OP_INPUT_EXIST(op, "input", data_type);
//   FORCE_OP_INPUT_EXIST(op, "weight", data_type);
//   FORCE_OP_INPUT_EXIST(op, "bias", data_type);

//   // _cpu_generic_compute(op, _cth_conv1d, data_type);
// }
