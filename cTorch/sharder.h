#ifndef CTH_SHARDER_H
#define CTH_SHARDER_H

#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Sharding an op's inputs & outputs for element-wise operation.
 *
 *  Params:
 *    - op: the target operator
 *    - n_shards: number of sharding pieces
 *
 *  Return:
 *    List of sharded operator.
 *
 *  Warning:
 *    The sharded operator needs to be taken care in terms of memory allocation.
 *  It may cause memory leak.
 */
List(CTorchOperator) *
    sharding_op_elewise(CTorchOperator *op, thread_n_t n_shards);

/**
 * Sharding a tensor for element-wise operator.
 *
 * Params:
 *  - tensor: target tensor
 *  - n_shards: total number of sharding pieces
 *  - tensor_list: sharded tensor will be inserted into this list
 */
void sharding_tensor_elewise(
    CTorchTensor *tensor,
    thread_n_t n_shards,
    List(CTorchTensor) * tensor_list);

#endif /* SHARDER_H */
