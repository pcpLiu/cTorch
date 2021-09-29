// Copyright 2021 Zhonghao Liu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_hardtanh_kernel(op, data_type)                                    \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
                                                                               \
    cth_tensor_dim_t N = input->meta_info->n_elements;                         \
                                                                               \
    data_type *input_ptr = (data_type *)input->values;                         \
    data_type *output_ptr = (data_type *)output->values;                       \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      if (input_ptr[i] > 1) {                                                  \
        output_ptr[i] = (data_type)1;                                          \
      } else if (input_ptr[i] < -1) {                                          \
        output_ptr[i] = (data_type)-1;                                         \
      } else {                                                                 \
        output_ptr[i] = (data_type)input_ptr[i];                               \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Applies the element-wise function hardtanh
 *
 * @param op CTHOperator operator
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor
 *   - # of output: 1
 *    - 0: output tensor
 */
void op_Hardtanh_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE data_type = input->meta_info->data_type;
  _cpu_generic_compute(op, _cth_hardtanh_kernel, data_type);
}
