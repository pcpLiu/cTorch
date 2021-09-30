/**
 * Copyright 2021 Zhonghao Liu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_prelu_kernel(op, data_type)                                       \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *weight = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);      \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
                                                                               \
    data_type *input_ptr = (data_type *)input->values;                         \
    data_type *weight_ptr = (data_type *)weight->values;                       \
    data_type *output_ptr = (data_type *)output->values;                       \
                                                                               \
    cth_tensor_dim_t *num_parameters;                                          \
    cth_extract_param_value(                                                   \
        op, CTH_PARAM_TYPE_NUM_PARAMETERS, (void **)&num_parameters, true);    \
                                                                               \
    FORCE_EQ(                                                                  \
        weight->meta_info->n_elements,                                         \
        PTR_VAL(num_parameters),                                               \
        "_cth_prelu_kernel dim error. num_parameters is %ld. weight tensor "   \
        "should only has same amount of elements but got %ld.",                \
        PTR_VAL(num_parameters),                                               \
        weight->meta_info->n_elements);                                        \
                                                                               \
    if (PTR_VAL(num_parameters) != 1) {                                        \
      FORCE_TRUE(                                                              \
          input->meta_info->n_dim >= 2,                                        \
          "_cth_prelu_kernel dim error. num_parameters is %ld. input "         \
          "tensor's n_dim should >=2, got %ld. ",                              \
          PTR_VAL(num_parameters),                                             \
          input->meta_info->n_dim);                                            \
                                                                               \
      FORCE_EQ(                                                                \
          input->meta_info->dims[1],                                           \
          PTR_VAL(num_parameters),                                             \
          "_cth_prelu_kernel dim error. num_parameters is %ld, which "         \
          "should equal to input tensor's dim[1] (%ld) or 1",                  \
          PTR_VAL(num_parameters),                                             \
          input->meta_info->dims[1]);                                          \
    }                                                                          \
                                                                               \
    if (PTR_VAL(num_parameters) == 1) {                                        \
      cth_tensor_dim_t N = input->meta_info->n_elements;                       \
      for (cth_tensor_dim_t i = 0; i < N; i++) {                               \
        output_ptr[i] =                                                        \
            (input_ptr[i] >= 0 ? input_ptr[i]                                  \
                               : (data_type)weight_ptr[0] * input_ptr[i]);     \
      }                                                                        \
    } else {                                                                   \
      cth_tensor_dim_t n_ele_channel = 1;                                      \
      cth_tensor_dim_t n_dim = input->meta_info->n_dim;                        \
      for (cth_tensor_dim_t i = 2; i < n_dim; i++) {                           \
        n_ele_channel *= input->meta_info->dims[i];                            \
      }                                                                        \
                                                                               \
      cth_tensor_dim_t n_ele_batch =                                           \
          n_ele_channel * input->meta_info->dims[1];                           \
                                                                               \
      cth_tensor_dim_t channels = input->meta_info->dims[1];                   \
      cth_tensor_dim_t batches = input->meta_info->dims[0];                    \
                                                                               \
      for (cth_tensor_dim_t b_i = 0; b_i < batches; b_i++) {                   \
        for (cth_tensor_dim_t c_i = 0; c_i < channels; c_i++) {                \
          cth_tensor_dim_t offset = b_i * n_ele_batch + c_i * n_ele_channel;   \
                                                                               \
          for (cth_tensor_dim_t i = 0; i < n_ele_channel; i++) {               \
            output_ptr[offset + i] =                                           \
                (input_ptr[offset + i] >= 0                                    \
                     ? input_ptr[offset + i]                                   \
                     : (data_type)weight_ptr[c_i] * input_ptr[offset + i]);    \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
  } while (0)

/**
 * @brief Applies the element-wise function PReLU
 *
 * @param op CTHOperator operator
 *
 * @note Channel dim is the 2nd dim of input. When input has dims < 2, then
 * there is no channel dim and the number of channels = 1.
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor
 *    - 1: weight tensor [num_parameters]
 *   - # of output: 1
 *    - 0: output tensor
 *
 * @par Op arguments:
 *    - 0: CTH_PARAM_TYPE_NUM_PARAMETERS num_parameters. number of aa to learn.
 *      Although it takes an int as input, there is only two values are
 *      legitimate: 1, or the number of channels at input. Default: 1
 */
void op_PReLU_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_NUM_PARAMETERS);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE data_type = input->meta_info->data_type;
  _cpu_generic_compute(op, _cth_prelu_kernel, data_type);
}
