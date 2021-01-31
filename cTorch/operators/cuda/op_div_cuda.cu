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

#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/operators/cuda/op_list_cuda.h"
#include "cTorch/operators/cuda/util_cuda.h"

#define _cth_div_cuda_atomic(x, y) (x / y)

_cth_declare_cuda_binary_kernel(cth_div_cuda, fdividef, _cth_div_cuda_atomic);

/**
 * @brief Divide two floating point values
 *
 * @param[CTHOperator] op operator
 *
 * @note CUDA onlly support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 2
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_div_cuda(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DEVICE device = input_1->meta_info->device;

  _cth_cuda_binary_workflow(
      input_1->meta_info->data_type,
      input_1->values,
      input_2->values,
      output->values,
      input_1->meta_info->n_elements,
      CTH_CUDA_THREADS_PER_BLOCK,
      cth_div_cuda,
      device);
}

#ifdef __cplusplus
}
#endif
