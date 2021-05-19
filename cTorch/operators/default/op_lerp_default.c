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

#define _cth_lerp_kernel(op, data_type)                                        \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *input_3 = cth_array_at(CTHTensor)(op->in_bound_tensors, 2);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *ptr_1 = (data_type *)input_1->values;                           \
    data_type *ptr_2 = (data_type *)input_2->values;                           \
    data_type *ptr_3 = (data_type *)input_3->values;                           \
    data_type *ptr_output = (data_type *)output->values;                       \
    cth_tensor_dim_t N = input_1->meta_info->n_elements;                       \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      ptr_output[i] = ptr_1[i] + ptr_3[i] * (ptr_2[i] - ptr_1[i]);             \
    }                                                                          \
  } while (0)

/**
 * out = input_1 + input_3 * (input_2 - input_1)
 *
 * Note: unlike PyTorch, no scalar is allowed here
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_lerp_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 3, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);
  // TODO: same type force

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_lerp_kernel, input->meta_info->data_type);
}
