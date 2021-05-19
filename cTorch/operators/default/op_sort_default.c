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
 * @brief Sorts the elements of the input tensor along a given dimension in
 * ascending order by value.
 *
 * If dim is not given, the last dimension of the input is chosen.
 *
 * If descending is True then the elements are sorted in descending order by
 * value.
 *
 * A namedtuple of (values, indices) is returned, where the values are the
 * sorted alues and indices are the indices of the elements in the original
 * input tensor.
 *
 * @note In this implementation, keepdim is always false.
 *
 * @param op
 *
 * Inputs & Outputs & Params:
 *    - # of inputs: 1
 *    - # of outputs: 2
 *      - 0: sorted tensor
 *      - 1: index
 *    - Argument:
 *      - dim (int): the dimension to reduce. Must be >= 0
 */
void op_sort_cpu(CTHOperator *op) {
  // TODO: imp
}
