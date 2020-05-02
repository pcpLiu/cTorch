#include "cTorch/sharder.h"

List(CTorchOperator) *
    sharding_op_elewise(CTorchOperator *op, thread_n_t n_shards) {
  // creating sharded ops
  List(CTorchOperator) *shards = new_list(CTorchOperator)();
  for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
    CTorchOperator *shard_op = MALLOC(sizeof(CTorchOperator));
    shard_op->op_id = op->op_id;
    shard_op->in_bound_tensors = new_list(CTorchTensor)();
    shard_op->out_bound_tensors = new_list(CTorchTensor)();
    insert_list(CTorchOperator)(shards, shard_op);
  }

  for (list_index_t tesnro_index = 0; tesnro_index < op->in_bound_tensors->size;
       tesnro_index++) {
    // sharding a input tensor, will get n_shards tensors
    List(CTorchTensor) *shared_tensors = new_list(CTorchTensor)();
    sharding_tensor_elewise(
        list_at(CTorchTensor)(op->in_bound_tensors, tesnro_index),
        n_shards,
        shared_tensors);

    // assign shared tensor to corresponding sharded op's tensor list
    for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = list_at(CTorchOperator)(shards, shard_i);
      insert_list(CTorchTensor)(
          shard_op->in_bound_tensors,
          list_at(CTorchTensor)(shared_tensors, shard_i));
    }
  }

  for (list_index_t tesnro_index = 0;
       tesnro_index < op->out_bound_tensors->size;
       tesnro_index++) {
    List(CTorchTensor) *shared_tensors = new_list(CTorchTensor)();
    sharding_tensor_elewise(
        list_at(CTorchTensor)(op->in_bound_tensors, tesnro_index),
        n_shards,
        shared_tensors);
    for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = list_at(CTorchOperator)(shards, shard_i);
      insert_list(CTorchTensor)(
          shard_op->out_bound_tensors,
          list_at(CTorchTensor)(shared_tensors, shard_i));
    }
  }

  return shards;
}

void sharding_tensor_elewise(
    CTorchTensor *tensor,
    thread_n_t n_shards,
    List(CTorchTensor) * tensor_list) {}
