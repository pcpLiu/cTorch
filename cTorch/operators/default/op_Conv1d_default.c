#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_conv1d(op, data_type)                                             \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *weight = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);      \
    CTHTensor *bias = cth_array_at(CTHTensor)(op->in_bound_tensors, 2);        \
                                                                               \
    cth_channel_t *in_channels, *out_channels;                                 \
    cth_kernel_t *kernel_size;                                                 \
    cth_groups_t *groups;                                                      \
    cth_stride_t *stride;                                                      \
    cth_pad_t *padding;                                                        \
    cth_dilation_t *dilation;                                                  \
    CTH_PADDING_MODE *padding_mode;                                            \
                                                                               \
    EXTRACT_PARAM_VALUE(                                                       \
        op, CTH_PARAM_TYPE_IN_CHANNELS, in_channels, in_channels);             \
    EXTRACT_PARAM_VALUE(                                                       \
        op, CTH_PARAM_TYPE_OUT_CHANNELS, out_channels, out_channels);          \
    EXTRACT_PARAM_VALUE(                                                       \
        op, CTH_PARAM_TYPE_KERNEL_SIZE, kernel_size_l1, kernel_size);          \
                                                                               \
    EXTRACT_PARAM_VALUE_OR_NULL(op, CTH_PARAM_TYPE_IN_GROUPS, groups, groups); \
    EXTRACT_PARAM_VALUE_OR_NULL(                                               \
        op, CTH_PARAM_TYPE_STRIDE_D2, stride_l2, stride);                      \
    EXTRACT_PARAM_VALUE_OR_NULL(                                               \
        op, CTH_PARAM_TYPE_PADDING_D2, padding_l2, padding);                   \
    EXTRACT_PARAM_VALUE_OR_NULL(                                               \
        op, CTH_PARAM_TYPE_DILATION_D2, dilation_l2, dilation);                \
    EXTRACT_PARAM_VALUE_OR_NULL(                                               \
        op, CTH_PARAM_TYPE_PADDING_MODE, padding_mode, padding_mode);          \
  } while (0);

/**
 * @brief Applies a 1D convolution over an input signal composed of several
 * input planes. A naive implementation.
 *
 * @param[CTHOperator] op operator
 *
 * @note Do not support boolean operator
 *
 * Inputs & outputs:
 *   - # of input: 3
 *    - 0: input tensor
 *    - 1: weight tensor [out_channels, in_channels/groups, kernel_size]
 *    - 2: bias tensor
 *   - # of output: 1
 */
void op_Conv1d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 3, 1);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_IN_CHANNELS);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_OUT_CHANNELS);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_KERNEL_SIZE_D2);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE data_type = input->meta_info->data_type;
  FORCE_OP_INPUT_EXIST(op, "input", data_type);
  FORCE_OP_INPUT_EXIST(op, "weight", data_type);
  FORCE_OP_INPUT_EXIST(op, "bias", data_type);

  _cpu_generic_compute(op, _cth_conv1d, data_type);
}
