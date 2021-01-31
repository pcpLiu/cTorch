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
 * @brief 1D replication padding.
 *
 * @param in_ptr input tensor value pointer
 * @param in_meta CTHTensorMeta input tensor meta
 * @param in_offset cth_tensor_dim_t input offset
 * @param out_ptr output tensor value pointer
 * @param out_meta CTHTensorMeta output tensor meta
 * @param out_offset cth_tensor_dim_t output offset
 * @param padding_left cth_tensor_dim_t left padding size
 * @param padding_right cth_tensor_dim_t right padding size
 */
#define _cth_replicate_pad_1d(                                                 \
    in_ptr,                                                                    \
    in_meta,                                                                   \
    in_offset,                                                                 \
    out_ptr,                                                                   \
    out_meta,                                                                  \
    out_offset,                                                                \
    padding_left,                                                              \
    padding_right)                                                             \
  do {                                                                         \
    cth_tensor_dim_t input_x_dim = in_meta->dims[2];                           \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                      \
      out_ptr[out_offset + i] = in_ptr[in_offset];                             \
    }                                                                          \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < padding_right; i++) {                     \
      out_ptr[out_offset + padding_left + input_x_dim + i] =                   \
          in_ptr[in_offset + input_x_dim - 1];                                 \
    }                                                                          \
  } while (0)

/**
 * @brief Pads the input tensor using replication of the input boundary.
 *
 * @param op CTHOperator operator
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor, [batch, channel (z), length (x)]
 *   - # of output: 1
 *    - 0: output tensor, [batch, channel (z), length (x)]
 *
 * @par Op arguments:
 *    - CTH_PARAM_TYPE_PADDING_D2
 */
void op_ReplicationPad1d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  _cth_padding_generic_1d(op, _cth_replicate_pad_1d);
}
