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

#define _cth_constant_pad_1d_extra_force()                                     \
  {}

/**
 * @brief 1D constant padding.
 * All the variables are defined in `_cth_padding_flow_1d`
 *
 */
#define _cth_constant_pad_1d(op, data_type)                                    \
  do {                                                                         \
    float *padding_value_float;                                                \
    EXTRACT_PARAM_VALUE(                                                       \
        op,                                                                    \
        CTH_PARAM_TYPE_PADDING_VALUE_FLOAT,                                    \
        padding_value_float,                                                   \
        padding_value_float);                                                  \
    data_type padding_value_typed = (data_type)padding_value_float[0];         \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                      \
      out_ptr[out_offset + i] = padding_value_typed;                           \
    }                                                                          \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < padding_right; i++) {                     \
      out_ptr[out_offset + padding_left + input_x_dim + i] =                   \
          padding_value_typed;                                                 \
    }                                                                          \
  } while (0)

/**
 * @brief Pads the input tensor boundaries with a constant value.
 *
 * @param op CTHOperator operator
 *
 * @note For int-like input tensor, this op will cast
 * `CTH_PARAM_TYPE_PADDING_VALUE_FLOAT` to corresponding integer type.
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor, [batch, channel (z), length (x)]
 *   - # of output: 1
 *    - 0: output tensor, [batch, channel (z), length (x)]
 *
 * @par Op arguments:
 *    - 0: CTH_PARAM_TYPE_PADDING_D2
 *    - 1: CTH_PARAM_TYPE_PADDING_VALUE_FLOAT
 */
void op_ConstantPad1d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  FORCE_OP_PARAM_NUM(op, 2);
  _cth_padding_generic_1d(
      op, _cth_constant_pad_1d, _cth_constant_pad_1d_extra_force);
}
