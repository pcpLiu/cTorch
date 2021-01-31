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

#include "cTorch/operators/apple/op_list_apple.h"
#include "cTorch/operators/apple/util_apple.h"
#include <Accelerate/Accelerate.h>

/**
 * @brief atan2(x, y)
 *
 * @note Apple backend only support float & double tensors
 *
 * Inputs & outputs:
 *    - # of input: 2
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_atan2_apple(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *in_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *in_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  int N = (int)in_1->meta_info->n_elements;
  // Accelerate use different inptu order
  _cth_apple_vforce_function_call_binary(
      in_1->meta_info->data_type,
      atan2,
      in_2->values,
      in_1->values,
      out->values,
      &N);
}
