#include "cTorch/sharder.h"

#include <tgmath.h>

void cth_sharding_op_elewise(
    CTorchOperator *op,
    cth_thread_n_t n_shards,
    CTHList(CTorchOperator) * ops) {

  /**
   * 1. Create sharded ops
   * 2. Prepare inbound & outbound tensors for each sharded ops
   *    2.1. Shard a tensor
   *    2.2. Assign each sharded piece to corresponding sharded op
   */

  for (cth_thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
    CTorchOperator *shard_op = MALLOC(sizeof(CTorchOperator));
    shard_op->op_id = op->op_id;
    shard_op->in_bound_tensors =
        new_array(CTorchTensor)(op->in_bound_tensors->size);
    shard_op->out_bound_tensors =
        new_array(CTorchTensor)(op->out_bound_tensors->size);
    shard_op->params = new_array(CTorchParam)(op->params->size);

    /* Copy params */
    array_index_t i = 0;
    while (i < op->params->size) {
      array_set(CTorchParam)(shard_op->params, i, MALLOC(sizeof(CTorchParam)));
      cth_copy_param(
          array_at(CTorchParam)(op->params, i),
          array_at(CTorchParam)(shard_op->params, i));
      i++;
    }

    cth_insert_list(CTorchOperator)(ops, shard_op);
  }

  for (array_index_t tensor_index = 0;
       tensor_index < op->in_bound_tensors->size;
       tensor_index++) {
    CTHList(CTorchTensor) *sharded_tensors = cth_new_list(CTorchTensor)();
    cth_sharding_tensor_elewise(
        array_at(CTorchTensor)(op->in_bound_tensors, tensor_index),
        n_shards,
        sharded_tensors);

    for (cth_thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = cth_list_at(CTorchOperator)(ops, shard_i);
      array_set(CTorchTensor)(
          shard_op->in_bound_tensors,
          tensor_index,
          cth_list_at(CTorchTensor)(sharded_tensors, shard_i));
    }

    cth_free_list(CTorchTensor)(sharded_tensors);
  }

  for (array_index_t tensor_index = 0;
       tensor_index < op->out_bound_tensors->size;
       tensor_index++) {
    CTHList(CTorchTensor) *sharded_tensors = cth_new_list(CTorchTensor)();
    cth_sharding_tensor_elewise(
        array_at(CTorchTensor)(op->out_bound_tensors, tensor_index),
        n_shards,
        sharded_tensors);

    for (cth_thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTorchOperator *shard_op = cth_list_at(CTorchOperator)(ops, shard_i);
      array_set(CTorchTensor)(
          shard_op->out_bound_tensors,
          tensor_index,
          cth_list_at(CTorchTensor)(sharded_tensors, shard_i));
    }

    // mem clean
    cth_free_list(CTorchTensor)(sharded_tensors);
  }
}

void cth_sharding_tensor_elewise(
    CTorchTensor *tensor,
    cth_thread_n_t n_shards,
    CTHList(CTorchTensor) * tensors) {

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

  for (cth_thread_n_t i = 0; i < n_shards; i++) {
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

    cth_insert_list(CTorchTensor)(tensors, shard_tensor);
  }
}
