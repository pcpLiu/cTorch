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

#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_atan2_kernel(op, data_type)                                       \
  do {                                                                         \
    CTHTensor *input_x = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_y = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *ptr_x = (data_type *)input_x->values;                           \
    data_type *ptr_y = (data_type *)input_y->values;                           \
    data_type *ptr_output = (data_type *)output->values;                       \
    cth_tensor_dim_t N = input_x->meta_info->n_elements;                       \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      ptr_output[i] = atan2(ptr_y[i], ptr_x[i]);                               \
    }                                                                          \
  } while (0)

/**
 * Computation see wiki: https://en.wikipedia.org/wiki/Atan2
 *
 * Inputs & outputs:
 *    - # of input: 2
 *        - 0: x
 *        - 1: y
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_atan2_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_atan2_kernel, input->meta_info->data_type);
}
