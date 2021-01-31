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

#define _cth_angle_kernel(input_ptr, output_ptr, N, data_type)                 \
  do {                                                                         \
    data_type *input_t = (data_type *)input_ptr;                               \
    data_type *output_t = (data_type *)output_ptr;                             \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = (data_type)((float)input_t[i] / CTH_PI);                   \
    }                                                                          \
  } while (0)

/**
 * Computes the element-wise angle (in radians) of the given input tensor.
 * Computation: input / Pi
 *
 * Note:
 * During computation, input values will be converted to float first and then
 * final result will be stored into output tensor as its datatype.
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_angle_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);
  // TODO: same type force

  CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  cth_tensor_dim_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary_generic(
      in->values, out->values, in->meta_info->data_type, N, _cth_angle_kernel);
}
