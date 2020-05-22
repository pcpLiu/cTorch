#include "cTorch/sharder.h"

#include <tgmath.h>

void cth_sharding_op_elewise(
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

  for (list_index_t tesnro_index = 0; tesnro_index < op->in_bound_tensors->size;
       tesnro_index++) {
    List(CTorchTensor) *sharded_tensors = new_list(CTorchTensor)();
    cth_sharding_tensor_elewise(
        list_at(CTorchTensor)(op->in_bound_tensors, tesnro_index),
        n_shards,
        sharded_tensors);

    for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = list_at(CTorchOperator)(ops, shard_i);
      insert_list(CTorchTensor)(
          shard_op->in_bound_tensors,
          list_at(CTorchTensor)(sharded_tensors, shard_i));
    }

    free_list(CTorchTensor)(sharded_tensors);
  }

  for (list_index_t tesnro_index = 0;
       tesnro_index < op->out_bound_tensors->size;
       tesnro_index++) {
    List(CTorchTensor) *sharded_tensors = new_list(CTorchTensor)();
    cth_sharding_tensor_elewise(
        list_at(CTorchTensor)(op->out_bound_tensors, tesnro_index),
        n_shards,
        sharded_tensors);

    for (thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = list_at(CTorchOperator)(ops, shard_i);
      insert_list(CTorchTensor)(
          shard_op->out_bound_tensors,
          list_at(CTorchTensor)(sharded_tensors, shard_i));
    }

    // mem clean
    free_list(CTorchTensor)(sharded_tensors);
  }
}

void cth_sharding_tensor_elewise(
    CTorchTensor *tensor, thread_n_t n_shards, List(CTorchTensor) * tensors) {

  /**
   * Shard a tensor into n_shards tensors evenly, except the last one which
   * will have all the rest values.
   *
   * Note:
   * When sharding a tensor, we will ignore it's dimension and view the
   * tensor as a flattned 1D tensor.
   */

  CTorchTensorMeta *raw_meta = tensor->meta_info;

  tensor_size_t n_elements =
      (tensor_size_t)floor((double)raw_meta->n_elements / (double)n_shards);
  tensor_size_t last_n_elements = n_elements + raw_meta->n_elements % n_shards;

  for (thread_n_t i = 0; i < n_shards; i++) {
    char *name = NULL;
    cth_asprintf(&name, "%s_shard_%u", tensor->meta_info->tensor_name, i);
    CTorchTensorMeta *meta = MALLOC(sizeof(CTorchTensorMeta));
    meta->value_size_of = raw_meta->value_size_of;
    meta->data_type = raw_meta->data_type;
    meta->is_sharded = true;
    meta->n_dim = 1;
    meta->dims = NULL; // elewise op does not use this field
    meta->align_size = raw_meta->align_size;
    meta->type = raw_meta->type;
    meta->n_elements = (n_shards - 1 == i ? last_n_elements : n_elements);
    meta->tensor_name = name;

    CTorchTensor *shard_tensor = MALLOC(sizeof(CTorchTensor));
    shard_tensor->meta_info = meta;
    shard_tensor->values = cth_tensor_ptr_offset(tensor, i * meta->n_elements);

    insert_list(CTorchTensor)(tensors, shard_tensor);
  }
}