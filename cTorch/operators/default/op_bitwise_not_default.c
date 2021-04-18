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

#define _cth_bitwise_not(op, data_type)                                        \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *in_ptr = (data_type *)input->values;                            \
    data_type *out_ptr = (data_type *)output->values;                          \
    cth_tensor_dim_t N = input->meta_info->n_elements;                         \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      out_ptr[i] = ~in_ptr[i];                                                 \
    }                                                                          \
  } while (0)

/**
 * Computes the bitwise NOT of the given input tensor.
 * The input tensor must be of integral or Boolean types. For bool tensors, it
 * computes the logical NOT.
 *
 * # of input: 1
 * # of output: 1
 */
void op_bitwise_not_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  cth_tensor_dim_t N = in->meta_info->n_elements;
  CTH_TENSOR_DATA_TYPE data_type = in->meta_info->data_type;

  CTH_TENSOR_DATA_TYPE types[5] = {
      CTH_TENSOR_DATA_TYPE_BOOL,
      CTH_TENSOR_DATA_TYPE_INT_16,
      CTH_TENSOR_DATA_TYPE_INT_32,
      CTH_TENSOR_DATA_TYPE_INT_64,
      CTH_TENSOR_DATA_TYPE_UINT_8,
  };
  CTH_FORCE_TENSOR_TYPES(in, types, 5);
  CTH_FORCE_TENSOR_TYPES(out, types, 5);

  _cpu_bit_compute(op, _cth_bitwise_not, data_type);
}
