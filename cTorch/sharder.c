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

#include "cTorch/sharder.h"

#include <tgmath.h>

void cth_sharding_op_elewise(
    CTHOperator *op, cth_thread_n_t n_shards, CTHList(CTHOperator) * ops) {

  /**
   * 1. Create sharded ops and copy all information from the original op
   * 2. Prepare inbound & outbound tensors for each sharded ops
   *    2.1. Shard a tensor
   *    2.2. Assign each sharded piece to corresponding sharded op
   */

  for (cth_thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
    CTHOperator *shard_op = MALLOC(sizeof(CTHOperator));
    shard_op->op_id = op->op_id;
    shard_op->in_bound_tensors =
        cth_new_array(CTHTensor)(op->in_bound_tensors->size);
    shard_op->out_bound_tensors =
        cth_new_array(CTHTensor)(op->out_bound_tensors->size);
    shard_op->params = cth_new_array(CTHParam)(op->params->size);

    cth_array_index_t i = 0;
    while (i < op->params->size) {
      cth_array_set(CTHParam)(shard_op->params, i, MALLOC(sizeof(CTHParam)));
      cth_copy_param(
          cth_array_at(CTHParam)(op->params, i),
          cth_array_at(CTHParam)(shard_op->params, i));
      i++;
    }

    cth_insert_list(CTHOperator)(ops, shard_op);
  }

  for (cth_array_index_t tensor_index = 0;
       tensor_index < op->in_bound_tensors->size;
       tensor_index++) {
    CTHList(CTHTensor) *sharded_tensors = cth_new_list(CTHTensor)();
    cth_sharding_tensor_elewise(
        cth_array_at(CTHTensor)(op->in_bound_tensors, tensor_index),
        n_shards,
        sharded_tensors);

    for (cth_thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTHOperator *shard_op = cth_list_at(CTHOperator)(ops, shard_i);
      cth_array_set(CTHTensor)(
          shard_op->in_bound_tensors,
          tensor_index,
          cth_list_at(CTHTensor)(sharded_tensors, shard_i));
    }

    cth_free_list(CTHTensor)(sharded_tensors);
  }

  for (cth_array_index_t tensor_index = 0;
       tensor_index < op->out_bound_tensors->size;
       tensor_index++) {
    CTHList(CTHTensor) *sharded_tensors = cth_new_list(CTHTensor)();
    cth_sharding_tensor_elewise(
        cth_array_at(CTHTensor)(op->out_bound_tensors, tensor_index),
        n_shards,
        sharded_tensors);

    for (cth_thread_n_t shard_i = 0; shard_i < n_shards; shard_i++) {
      CTHOperator *shard_op = cth_list_at(CTHOperator)(ops, shard_i);
      cth_array_set(CTHTensor)(
          shard_op->out_bound_tensors,
          tensor_index,
          cth_list_at(CTHTensor)(sharded_tensors, shard_i));
    }

    // mem clean
    cth_free_list(CTHTensor)(sharded_tensors);
  }
}

void cth_sharding_tensor_elewise(
    CTHTensor *tensor, cth_thread_n_t n_shards, CTHList(CTHTensor) * tensors) {

  /**
   * Shard a tensor into n_shards tensors evenly, except the last one which
   * will have all the rest values.
   *
   * Note:
   * When sharding a tensor, we will ignore it's dimension and view the
   * tensor as a flattned 1D tensor.
   */

  CTHTensorMeta *raw_meta = tensor->meta_info;

  cth_tensor_dim_t n_elements =
      (cth_tensor_dim_t)floor((double)raw_meta->n_elements / (double)n_shards);
  cth_tensor_dim_t last_n_elements =
      n_elements + raw_meta->n_elements % n_shards;

  for (cth_thread_n_t i = 0; i < n_shards; i++) {
    char *name = NULL;
    cth_asprintf(&name, "%s_shard_%u", tensor->meta_info->tensor_name, i);
    CTHTensorMeta *meta = MALLOC(sizeof(CTHTensorMeta));
    meta->value_size_of = raw_meta->value_size_of;
    meta->data_type = raw_meta->data_type;
    meta->is_sharded = true;
    meta->n_dim = 1;
    meta->dims = NULL; // elewise op does not use this field
    meta->align_size = raw_meta->align_size;
    meta->type = raw_meta->type;
    meta->n_elements = (n_shards - 1 == i ? last_n_elements : n_elements);
    meta->tensor_name = name;

    CTHTensor *shard_tensor = MALLOC(sizeof(CTHTensor));
    shard_tensor->meta_info = meta;
    shard_tensor->values = cth_tensor_ptr_offset(tensor, i * meta->n_elements);

    cth_insert_list(CTHTensor)(tensors, shard_tensor);
  }
}
