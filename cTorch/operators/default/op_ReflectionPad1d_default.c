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
#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * @brief Refelction has specfic argument force
 *
 */
#define _cth_reflection_pad_1d_extra_force()                                   \
  do {                                                                         \
    FORCE_TRUE(                                                                \
        padding_left < input_x_dim,                                            \
        "op_ReflectionPad1d_cpu force failed. padding_left (%ld) < "           \
        "input_x_dim (%ld) is false.",                                         \
        padding_left,                                                          \
        input_x_dim);                                                          \
                                                                               \
    FORCE_TRUE(                                                                \
        padding_right < input_x_dim,                                           \
        "op_ReflectionPad1d_cpu force failed. padding_right (%ld) < "          \
        "input_x_dim (%ld) is false.",                                         \
        padding_right,                                                         \
        input_x_dim);                                                          \
  } while (0)

/**
 * @brief 1D replication padding.
 * All the variables are defined in `_cth_padding_flow_1d`.
 */
#define _cth_reflection_pad_1d(op, data_type)                                  \
  do {                                                                         \
    for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                      \
      out_ptr[out_offset + padding_left - 1 - i] = in_ptr[in_offset + 1 + i];  \
    }                                                                          \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < padding_right; i++) {                     \
      out_ptr[out_offset + padding_left + input_x_dim + i] =                   \
          in_ptr[in_offset + input_x_dim - 2 - i];                             \
    }                                                                          \
  } while (0)

/**
 * @brief Pads the input tensor using the reflection of the input boundary.
 *
 * @param op CTHOperator operator
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor, [batch (b), channel (z), length (x)]
 *   - # of output: 1
 *    - 0: output tensor, [batch (b), channel (z), length (x)]
 *
 * @par Op arguments:
 *    - CTH_PARAM_TYPE_PADDING_D2
 */
void op_ReflectionPad1d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_PADDING_D2);
  _cth_padding_generic_1d(
      op, _cth_reflection_pad_1d, _cth_reflection_pad_1d_extra_force);
}
