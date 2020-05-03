#ifndef CTH_SHARDER_H
#define CTH_SHARDER_H

#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Sharding an op's inputs & outputs for element-wise operation.
 *
 * Params:
 *  - op: the target operator
 *  - n_shards: number of sharding pieces
 *  - ops: sharded operator list to be appended
 *
 * Return:
 *    List of sharded operator
 */

void sharding_op_elewise(
    CTorchOperator *op, thread_n_t n_shards, List(CTorchOperator) * ops);

/**
 * Sharding a tensor for element-wise operator.
 *
 * Params:
 *  - tensor: target tensor
 *  - n_shards: total number of sharding pieces
 *  - tensors: sharded tensor list to be appended
 *
 * Return:
 *  List of sharded tensors
 */
void sharding_tensor_elewise(
    CTorchTensor *tensor, thread_n_t n_shards, List(CTorchTensor) * tensors);

#endif /* SHARDER_H */
