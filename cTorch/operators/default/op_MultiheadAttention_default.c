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

/**
 * @brief MultiheadAttention operator.
 *
 * @note This is a composite opreator. It has dependcies on following ops:
 *   - op_Linear_cpu
 *   - op_Softmax_cpu
 *   - op_scale_cpu
 *   - op_matmul_cpu
 *
 * @param op CTHOperator operator
 */
void op_MultiheadAttention_cpu(CTHOperator *op) {}
