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
#include <tgmath.h>

#define _cth_logsumexp(                                                        \
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
    double result = 0;                                                         \
    for (cth_tensor_dim_t i = 0; i < reduce_size; i++) {                       \
      result += exp((double)in_ptr[start_offset + i * inner_offset]);          \
    }                                                                          \
                                                                               \
    if (output_dtype_enum == CTH_TENSOR_DATA_TYPE_INT_16 ||                    \
        output_dtype_enum == CTH_TENSOR_DATA_TYPE_INT_32 ||                    \
        output_dtype_enum == CTH_TENSOR_DATA_TYPE_INT_64 ||                    \
        output_dtype_enum == CTH_TENSOR_DATA_TYPE_UINT_8) {                    \
      out_ptr[result_offset] = round(log(result));                             \
    } else {                                                                   \
      out_ptr[result_offset] = log(result);                                    \
    }                                                                          \
  } while (0)

/**
 * @brief Returns the log of summed exponentials of each row of the input
 tensor
 * in the given dimension dim. The computation is numerically stabilized.
 *
 * @par Op requirement:
 *    - # of input tensors: 1
 *    - # of arguments: 1
 *        - CTH_PARAM_TYPE_DIM
 *    - # of output tensors: 1
 *        - The output tensor data type should be floating
 *
 * @note keepdim is always false as it setup in PyTorch
 */
void op_logsumexp_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  FORCE_OP_PARAM_NUM(op, 1);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_DIM);

  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  _cpu_reduce_dim_generic(
      op, in->meta_info->data_type, out->meta_info->data_type, _cth_logsumexp);
}
