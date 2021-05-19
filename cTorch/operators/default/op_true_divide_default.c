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

#define _cth_true_divide(op, data_type)                                        \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    float *out_ptr = (float *)output->values;                                  \
    cth_tensor_dim_t N = input_1->meta_info->n_elements;                       \
                                                                               \
    uint16_t x, y;                                                             \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      out_ptr[i] = (float)in_ptr_1[i] / (float)in_ptr_2[i];                    \
    }                                                                          \
  } while (0)

/**
 * @brief Performs “true division” that always computes the division in floating
 * point. Analogous to division in Python 3 and equivalent to torch.div() except
 * when both inputs have bool or integer scalar types, in which case they are
 * cast to the default (floating) scalar type before the division.
 *
 * @note Output tensor should be CTH_TENSOR_DATA_TYPE_FLOAT_32
 *
 * Inputs and outputs:
 *    - # of input: 2
 *      - 0: dividend
 *      - 1: divisor
 *    - # of output: 1
 */
void op_true_divide_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE types[1] = {CTH_TENSOR_DATA_TYPE_FLOAT_32};
  CTH_FORCE_TENSOR_TYPES(output, types, 1);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_true_divide, input->meta_info->data_type);
}
