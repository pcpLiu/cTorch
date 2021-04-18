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

#define _cth_bitwise_and(op, data_type)                                        \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    data_type *out_ptr = (data_type *)output->values;                          \
    cth_tensor_dim_t N = input_1->meta_info->n_elements;                       \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      out_ptr[i] = in_ptr_1[i] & in_ptr_2[i];                                  \
    }                                                                          \
  } while (0)

/**
 * Computes the bitwise AND of the given two input tensor.
 * The input tensor must be of integral or Boolean types. For bool tensors, it
 * computes the logical AND.
 *
 * # of input: 2
 * # of output: 1
 */
void op_bitwise_and_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  CTHTensor *in_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *in_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  cth_tensor_dim_t N = in_1->meta_info->n_elements;
  CTH_TENSOR_DATA_TYPE data_type = in_1->meta_info->data_type;

  CTH_TENSOR_DATA_TYPE types[5] = {
      CTH_TENSOR_DATA_TYPE_BOOL,
      CTH_TENSOR_DATA_TYPE_INT_16,
      CTH_TENSOR_DATA_TYPE_INT_32,
      CTH_TENSOR_DATA_TYPE_INT_64,
      CTH_TENSOR_DATA_TYPE_UINT_8,
  };
  CTH_FORCE_TENSOR_TYPES(in_1, types, 5);
  CTH_FORCE_TENSOR_TYPES(in_2, types, 5);
  CTH_FORCE_TENSOR_TYPES(out, types, 5);

  _cpu_bit_compute(op, _cth_bitwise_and, data_type);
}
