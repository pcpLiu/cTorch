#include "cTorch/operator.h"

impl_new_new_list_item_func(CTorchOperator);
impl_new_list_func(CTorchOperator);
impl_insert_list_func(CTorchOperator);

void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTorchOperator *op) {
  if (op->in_bound_tensors->size != op->out_bound_tensors->size) {
    FAIL_EXIT(CTH_LOG_STR,
              "Operator should have same numbers of input and output tensors.");
  }
}

void OP_FAIL_ON_DTYPE(CTorchOperator *op, CTH_TENSOR_DATA_TYPE data_type) {
  ListItem(CTorchTensor) *tensor_it = op->in_bound_tensors->head;

  for (uint32_t i = 0; i < op->in_bound_tensors->size; i++) {
    CTorchTensor *tensor = tensor_it->data;
    if (tensor->meta_info->data_type == data_type) {
      FAIL_EXIT(CTH_LOG_STR, "Operator does not support data type.");
    }
    tensor_it = tensor_it->next_item;
  }

  tensor_it = op->out_bound_tensors->head;
  for (uint32_t i = 0; i < op->out_bound_tensors->size; i++) {
    CTorchTensor *tensor = tensor_it->data;
    if (tensor->meta_info->data_type == data_type) {
      // TODO: better logging
      FAIL_EXIT(CTH_LOG_STR, "Operator does not support data type.");
    }
    tensor_it = tensor_it->next_item;
  }
}

void FORCE_OP_PARAM_EXIST(CTorchOperator *op,
                          const char *name,
                          CTH_TENSOR_DATA_TYPE data_type) {
  ListItem(CTorchTensor) *item = op->in_bound_tensors->head;
  bool found = false;
  for (int i = 0; i < op->in_bound_tensors->size; i++) {
    if (strcmp(name, item->data->meta_info->tensor_name) == 0 &&
        item->data->meta_info->data_type == data_type) {
      found = true;
      break;
    }
    item = item->next_item;
  }

  if (!found) {
    // TODO: better logging
    FAIL_EXIT(CTH_LOG_STR, "FORCE_OP_PARAM_EXIST failes.");
  }
}

CTorchTensor *_get_tensor_by_name(List(CTorchTensor) * tensor_list,
                                  const char *name,
                                  bool fail_exit) {
  CTorchTensor *tensor = NULL;
  ListItem(CTorchTensor) *item = tensor_list->head;
  for (int i = 0; i < tensor_list->size; i++) {
    if (strcmp(item->data->meta_info->tensor_name, name) == 0) {
      tensor = item->data;
      break;
    }
    item = item->next_item;
  }

  if (tensor == NULL && fail_exit) {
    FAIL_EXIT(CTH_LOG_STR, "Could not find tensor %s", name);
  } else {
    return tensor;
  }
}

CTorchTensor *
get_input_by_name(CTorchOperator *op, const char *name, bool fail_exit) {
  return _get_tensor_by_name(op->in_bound_tensors, name, true);
}

CTorchTensor *
get_output_by_name(CTorchOperator *op, const char *name, bool fail_exit) {
  return _get_tensor_by_name(op->out_bound_tensors, name, true);
}

CTorchOperator *_sharding_tensor_elewise(CTorchTensor *tensor,
                                         thread_n_t shard_index,
                                         thread_n_t n_shards) {}

List(CTorchOperator) *
    sharding_op_elewise(CTorchOperator *op, thread_n_t n_shards) {
  List(CTorchOperator) *shards = new_list(CTorchOperator)();

  for (thread_n_t i = 0; i < n_shards; i++) {
    CTorchOperator *shard_op = MALLOC(sizeof(CTorchOperator));
    shard_op->op_id = op->op_id;
    shard_op->in_bound_tensors = new_list(CTorchTensor)();
    shard_op->out_bound_tensors = new_list(CTorchTensor)();
    insert_list(CTorchOperator)(shards, shard_op);

    for (thread_n_t j = 0; j < op->in_bound_tensors->size; j++) {
    }
  }

  return shards;
}