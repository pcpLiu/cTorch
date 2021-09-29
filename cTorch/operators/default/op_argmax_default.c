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

#define _cth_argmax(                                                           \
    in_ptr,                                                                    \
    out_ptr,                                                                   \
    input_data_type,                                                           \
    output_data_type,                                                          \
    input_dtype_enum,                                                          \
    output_dtype_enum,                                                         \
    start_offset,                                                              \
    inner_offset,                                                              \
    result_offset,                                                             \
    reduce_size)                                                               \
  do {                                                                         \
    cth_tensor_dim_t max_i = 0;                                                \
    input_data_type max_val = in_ptr[start_offset];                            \
    for (cth_tensor_dim_t i = 0; i < reduce_size; i++) {                       \
      input_data_type val = in_ptr[start_offset + i * inner_offset];           \
      if (val > max_val) {                                                     \
        max_val = val;                                                         \
        max_i = i;                                                             \
      }                                                                        \
    }                                                                          \
    out_ptr[result_offset] = max_i;                                            \
  } while (0)

/**
 * @brief Returns the indices of the maximum values of a tensor across a
 * dimension.
 *
 * @par When there's multiple max values in a dim, it returns the last position
 * it meets.
 *
 * @note In this implementation, keepdim is always false.
 *
 * @param op
 *
 * Inputs & Outputs & Params:
 *    - # of inputs: 1
 *    - # of outputs: 1
 *      - Output tensor type should be `CTH_TENSOR_DATA_TYPE_INT_64`
 *    - Argument:
 *      - dim (int): the dimension to reduce. If `-1`, the argmax of
 *        the flattened input is returned.
 */
void op_argmax_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  FORCE_OP_PARAM_NUM(op, 1);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_DIM);

  CTH_TENSOR_DATA_TYPE types[1] = {
      CTH_TENSOR_DATA_TYPE_INT_64,
  };
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTH_FORCE_TENSOR_TYPES(out, types, 1);

  CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);

  _cpu_reduce_dim_generic(
      op, in->meta_info->data_type, out->meta_info->data_type, _cth_argmax);
}
