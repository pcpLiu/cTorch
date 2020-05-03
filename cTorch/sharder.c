#include "cTorch/sharder.h"

void sharding_op_elewise(
    CTorchOperator *op, thread_n_t n_shards, List(CTorchOperator) * ops) {

  /**
   * 1. Create sharded ops
   * 2. Prepare inbound & outbound tensors for each sharded ops
   *    2.1. Shard a tensor
   *    2.2. Assign each sharded piece to corresponding sharded op
   */

  for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
    CTorchOperator *shard_op = MALLOC(sizeof(CTorchOperator));
    shard_op->op_id = op->op_id;
    shard_op->in_bound_tensors = new_list(CTorchTensor)();
    shard_op->out_bound_tensors = new_list(CTorchTensor)();
    insert_list(CTorchOperator)(ops, shard_op);
  }

  // inboud
  for (list_index_t tesnro_index = 0; tesnro_index < op->in_bound_tensors->size;
       tesnro_index++) {
    // shard tensor
    List(CTorchTensor) *sharded_tensors = new_list(CTorchTensor)();
    sharding_tensor_elewise(
        list_at(CTorchTensor)(op->in_bound_tensors, tesnro_index),
        n_shards,
        sharded_tensors);

    // assign to sharded ops
    for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = list_at(CTorchOperator)(ops, shard_i);
      insert_list(CTorchTensor)(
          shard_op->in_bound_tensors,
          list_at(CTorchTensor)(sharded_tensors, shard_i));
    }

    free_list(CTorchTensor)(sharded_tensors);
  }

  // outbound
  for (list_index_t tesnro_index = 0;
       tesnro_index < op->out_bound_tensors->size;
       tesnro_index++) {
    List(CTorchTensor) *sharded_tensors = new_list(CTorchTensor)();
    sharding_tensor_elewise(
        list_at(CTorchTensor)(op->out_bound_tensors, tesnro_index),
        n_shards,
        sharded_tensors);

    for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = list_at(CTorchOperator)(ops, shard_i);
      insert_list(CTorchTensor)(
          shard_op->out_bound_tensors,
          list_at(CTorchTensor)(sharded_tensors, shard_i));
    }

    free_list(CTorchTensor)(sharded_tensors);
  }
}

void sharding_tensor_elewise(
    CTorchTensor *tensor, thread_n_t n_shards, List(CTorchTensor) * tensors) {}