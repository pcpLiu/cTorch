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

#define _cth_clamp(op, data_type)                                              \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    data_type *input_ptr = (data_type *)input->values;                         \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *output_ptr = (data_type *)output->values;                       \
    float *min, *max;                                                          \
    cth_extract_param_value(op, CTH_PARAM_TYPE_MIN, (void **)&min, true);      \
    cth_extract_param_value(op, CTH_PARAM_TYPE_MAX, (void **)&max, true);      \
    cth_tensor_dim_t N = input->meta_info->n_elements;                         \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      if (input_ptr[i] < *min) {                                               \
        output_ptr[i] = *min;                                                  \
      } else if (input_ptr[i] > *max) {                                        \
        output_ptr[i] = *max;                                                  \
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
 *        - CTH_PARAM_TYPE_MIN
 *        - CTH_PARAM_TYPE_MAX
 *    - # of output tensors: 1
 *        - output:  the input always at index 0
 *
 * Note: does not support boradcast
 */
void op_clamp_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  FORCE_OP_PARAM_NUM(op, 2);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_MAX);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_MIN);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE data_type = input->meta_info->data_type;

  _cpu_generic_compute(op, _cth_clamp, data_type);
}
