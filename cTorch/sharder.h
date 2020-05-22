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
    CTorchOperator *op, thread_n_t n_shards, List(CTorchOperator) * ops);

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
    CTorchTensor *tensor, thread_n_t n_shards, List(CTorchTensor) * tensors);

#endif /* SHARDER_H */
