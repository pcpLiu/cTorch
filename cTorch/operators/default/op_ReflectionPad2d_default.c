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
#define _cth_reflection_pad_2d_extra_force()                                   \
  do {                                                                         \
    FORCE_TRUE(                                                                \
        padding_left < in_x_dim,                                               \
        "op_ReflectionPad2d_cpu force failed. padding_left (%ld) < "           \
        "input_x_dim (%ld) is false.",                                         \
        padding_left,                                                          \
        in_x_dim);                                                             \
                                                                               \
    FORCE_TRUE(                                                                \
        padding_right < in_x_dim,                                              \
        "op_ReflectionPad2d_cpu force failed. padding_right (%ld) < "          \
        "input_x_dim (%ld) is false.",                                         \
        padding_right,                                                         \
        in_x_dim);                                                             \
  } while (0)

/**
 * @brief 2D replication padding.
 *
 * When dealing with `padding_whole_row`, we first locate the reflective
 * input y row, i.e. calculating `y_off_set`. Then we first fill in the
 * overlapped elements for output. We simply find correponding element in
 * reflective row. For left & right padding part, we use same reflection logic
 * but read data from just padded output.
 *
 * Note that `in_offset` is calcualted differently in `_cth_padding_flow_2d` for
 * top padding and bottom padding. In padding top, `in_offset` is the first row
 * of input. In padding bottom, `in_offset` is last row of input.
 *
 * All the variables are defined in `_cth_padding_flow_2d`
 *
 */
#define _cth_reflection_pad_2d(op, data_type)                                  \
  do {                                                                         \
    if (padding_whole_row) {                                                   \
      cth_tensor_dim_t y_offset = 0;                                           \
      if (y_i < padding_top) {                                                 \
        y_offset = in_x_dim * (padding_top - y_i);                             \
      } else {                                                                 \
        y_offset = in_x_dim * (y_i - padding_top - in_y_dim + 1);              \
        y_offset *= -1;                                                        \
      }                                                                        \
                                                                               \
      for (cth_tensor_dim_t i = 0; i < in_x_dim; i++) {                        \
        out_ptr[out_offset + padding_left + i] =                               \
            in_ptr[in_offset + i + y_offset];                                  \
      }                                                                        \
                                                                               \
      for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                    \
        out_ptr[out_offset + padding_left - 1 - i] =                           \
            out_ptr[out_offset + padding_left + 1 + i];                        \
      }                                                                        \
                                                                               \
      for (cth_tensor_dim_t i = 0; i < padding_right; i++) {                   \
        out_ptr[out_offset + padding_left + in_x_dim + i] =                    \
            out_ptr[out_offset + padding_left + in_x_dim - 2 - i];             \
      }                                                                        \
    } else {                                                                   \
      for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                    \
        out_ptr[out_offset + padding_left - 1 - i] =                           \
            in_ptr[in_offset + 1 + i];                                         \
      }                                                                        \
                                                                               \
      for (cth_tensor_dim_t i = 0; i < padding_right; i++) {                   \
        out_ptr[out_offset + padding_left + in_x_dim + i] =                    \
            in_ptr[in_offset + in_x_dim - 2 - i];                              \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Pads the input tensor using the reflection of the input boundary.
 *
 * @param op CTHOperator operator
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor, [batch (b), channel (c), height (y), length (x)]
 *   - # of output: 1
 *    - 0: output tensor, [batch (b), channel (c), height (y), length (x)]
 *
 * @par Op arguments:
 *    - CTH_PARAM_TYPE_PADDING_D4, [left, right, top, bottom]
 */
void op_ReflectionPad2d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  // FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_PADDING_D4);
  _cth_padding_generic_2d(
      op, _cth_reflection_pad_2d, _cth_reflection_pad_2d_extra_force);
}
