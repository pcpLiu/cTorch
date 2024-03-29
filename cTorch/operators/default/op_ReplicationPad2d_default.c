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

#define _cth_replicate_pad_2d_extra_force()                                    \
  {}

/**
 * @brief 2D replication padding.
 *
 * All the variables are defined in `_cth_padding_flow_2d`
 *
 */
#define _cth_replicate_pad_2d(op, data_type)                                   \
  do {                                                                         \
    for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                      \
      out_ptr[out_offset + i] = in_ptr[in_offset];                             \
    }                                                                          \
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
 * @brief Pads the input tensor using replication of the input boundary.
 *
 * @param op CTHOperator operator
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor, [batch (b), channel (c), height (y), length (x)]
 *   - # of output: 1
 *    - 0: output tensor, [batch (b), channel (), height (y), length (x)]
 *
 * @par Op arguments:
 *    - CTH_PARAM_TYPE_PADDING_D4, [left, right, top, bottom]
 */
void op_ReplicationPad2d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  _cth_padding_generic_2d(
      op, _cth_replicate_pad_2d, _cth_replicate_pad_2d_extra_force);
}
