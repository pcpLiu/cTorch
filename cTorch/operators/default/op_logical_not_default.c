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

#define _cth_logical_not(op, data_type)                                        \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *in_ptr = (data_type *)input->values;                            \
    bool *out_ptr = (bool *)output->values;                                    \
    cth_tensor_dim_t N = input->meta_info->n_elements;                         \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      out_ptr[i] = (bool)(1 - (in_ptr[i] == 0 ? 0 : 1));                       \
    }                                                                          \
  } while (0)

/**
 * Computes the element-wise logical NOT of the given input tensor. If not
 * specified, the output tensor will have the bool dtype. If the input tensor is
 * not a bool tensor, zeros are treated as False and non-zeros are treated as
 * True.
 *
 *
 * # of input: 2
 * # of output: 1
 *    - Must be bool type
 */
void op_logical_not_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);

  CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE types[1] = {CTH_TENSOR_DATA_TYPE_BOOL};
  CTH_FORCE_TENSOR_TYPES(output, types, 1);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_logical_not, input->meta_info->data_type);
}
