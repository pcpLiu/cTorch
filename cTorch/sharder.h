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

#ifndef CTH_SHARDER_H
#define CTH_SHARDER_H

#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Sharding an op's inputs & outputs for element-wise operation
 *
 * New memory allocation:
 *  - A list of sharded input & output tensors
 *  - A list of sharded ops
 *
 * Params:
 *  - op: the target operator
 *  - n_shards: number of sharding pieces
 *  - ops: sharded operator list to be appended
 *
 * Return:
 *    List of sharded operator
 */

void cth_sharding_op_elewise(
    CTHOperator *op, cth_thread_n_t n_shards, CTHList(CTHOperator) * ops);

/**
 * Sharding a tensor for element-wise operator
 *
 * New memory allocation:
 *  - A list of sharded tensors
 *
 * Params:
 *  - tensor: target tensor
 *  - n_shards: total number of sharding pieces
 *  - tensors: sharded tensor list to be appended
 *
 * Return:
 *  List of sharded tensors
 */
void cth_sharding_tensor_elewise(
    CTHTensor *tensor, cth_thread_n_t n_shards, CTHList(CTHTensor) * tensors);

#endif /* SHARDER_H */
