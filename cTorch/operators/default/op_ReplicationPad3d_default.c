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
 * @brief 3D replication padding on overlapping part.
 *
 * All variables defined `_cth_padding_flow_3d`.
 */
#define _cth_replicate_pad_3d()                                                \
  do {                                                                         \
    for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                      \
      out_ptr[out_offset + i] = in_ptr[in_offset];                             \
    }                                                                          \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < padding_right; i++) {                     \
      out_ptr[out_offset + padding_left + in_x_dim + i] =                      \
          in_ptr[in_offset + in_x_dim - 1];                                    \
    }                                                                          \
                                                                               \
    if (padding_whole_row) {                                                   \
      for (cth_tensor_dim_t i = 0; i < in_x_dim; i++) {                        \
        out_ptr[out_offset + padding_left + i] = in_ptr[in_offset + i];        \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief 3D replication padding on front part.
 * Just copy whole frame from the original input's 1st frame.
 *
 * All variables defined `_cth_padding_flow_3d`.
 */
#define _cth_replicate_pad_3d_front()                                          \
  do {                                                                         \
    cth_tensor_dim_t cp_src_offset =                                           \
        b_i * (out_c_dim * out_z_dim * out_y_dim * out_x_dim) +                \
        c_i * (out_z_dim * out_y_dim * out_x_dim) +                            \
        padding_front * (out_y_dim * out_x_dim);                               \
    cth_tensor_dim_t cp_ele_size = out_y_dim * out_x_dim;                      \
    for (cth_tensor_dim_t z_i = 0; z_i < padding_front; z_i++) {               \
      cth_tensor_dim_t out_offset =                                            \
          b_i * (out_c_dim * out_z_dim * out_y_dim * out_x_dim) +              \
          c_i * (out_z_dim * out_y_dim * out_x_dim) +                          \
          z_i * (out_y_dim * out_x_dim);                                       \
      memcpy(                                                                  \
          out_ptr + out_offset,                                                \
          out_ptr + cp_src_offset,                                             \
          data_size * cp_ele_size);                                            \
    }                                                                          \
  } while (0)

/**
 * @brief 3D replication padding on back part.
 * Just copy whole frame from the original input's last frame.
 *
 * All variables defined `_cth_padding_flow_3d`.
 */
#define _cth_replicate_pad_3d_back()                                           \
  do {                                                                         \
    cth_tensor_dim_t cp_src_offset =                                           \
        b_i * (out_c_dim * out_z_dim * out_y_dim * out_x_dim) +                \
        c_i * (out_z_dim * out_y_dim * out_x_dim) +                            \
        (out_z_dim - padding_back - 1) * (out_y_dim * out_x_dim);              \
    cth_tensor_dim_t cp_ele_size = out_y_dim * out_x_dim;                      \
    for (cth_tensor_dim_t z_i = out_z_dim - padding_back; z_i < out_z_dim;     \
         z_i++) {                                                              \
      cth_tensor_dim_t out_offset =                                            \
          b_i * (out_c_dim * out_z_dim * out_y_dim * out_x_dim) +              \
          c_i * (out_z_dim * out_y_dim * out_x_dim) +                          \
          z_i * (out_y_dim * out_x_dim);                                       \
      memcpy(                                                                  \
          out_ptr + out_offset,                                                \
          out_ptr + cp_src_offset,                                             \
          data_size * cp_ele_size);                                            \
    }                                                                          \
  } while (0)

/**
 * @brief Pads the input tensor using replication of the input boundary.
 *
 * @param op CTHOperator operator
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor, [batch (b), channel (c), depth (z), height (y),
 length
 * (x)]
 *   - # of output: 1
 *    - 0: output tensor, [batch (b), channel (c), depth (z), height (y),
 length
 * (x)]
 *
 * @par Op arguments:
 *    - CTH_PARAM_TYPE_PADDING_D6, [left, right, top, bottom, front, back]
 */
void op_ReplicationPad3d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  _cth_padding_generic_3d(
      op,
      _cth_replicate_pad_3d,
      _cth_replicate_pad_3d_front,
      _cth_replicate_pad_3d_back);
}
