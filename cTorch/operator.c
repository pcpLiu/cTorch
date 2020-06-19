#include "cTorch/operator.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <string.h>

impl_new_list_item_func(CTorchOperator);
impl_new_list_func(CTorchOperator);
impl_insert_list_func(CTorchOperator);
impl_list_at_func(CTorchOperator);
impl_list_pop_func(CTorchOperator);
impl_free_list_func(CTorchOperator);
impl_free_list_deep_func(CTorchOperator);

impl_new_array_func(CTorchTensor);
impl_array_at_func(CTorchTensor);
impl_array_set_func(CTorchTensor);

void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTorchOperator *op) {
  if (op->in_bound_tensors->size != op->out_bound_tensors->size) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "Operator should have same numbers of input and output tensors.");
  }
}

void OP_FAIL_ON_DTYPE(CTorchOperator *op, CTH_TENSOR_DATA_TYPE data_type) {
  CTorchTensor *tensor = NULL;
  for (array_index_t i = 0; i < op->in_bound_tensors->size; i++) {
    tensor = array_at(CTorchTensor)(op->in_bound_tensors, i);
    if (tensor->meta_info->data_type == data_type) {
      FAIL_EXIT(CTH_LOG_ERR, "Operator does not support data type.");
    }
  }

  for (array_index_t i = 0; i < op->out_bound_tensors->size; i++) {
    tensor = array_at(CTorchTensor)(op->in_bound_tensors, i);
    if (tensor->meta_info->data_type == data_type) {
      FAIL_EXIT(CTH_LOG_ERR, "Operator does not support data type.");
    }
  }
}

void FORCE_OP_PARAM_EXIST(
    CTorchOperator *op, const char *name, CTH_TENSOR_DATA_TYPE data_type) {
  CTorchTensor *tensor = NULL;
  bool found = false;
  for (array_index_t i = 0; i < op->in_bound_tensors->size; i++) {
    tensor = array_at(CTorchTensor)(op->in_bound_tensors, i);
    if (strcmp(name, tensor->meta_info->tensor_name) == 0 &&
        tensor->meta_info->data_type == data_type) {
      found = true;
      break;
    }
  }

  if (!found) {
    FAIL_EXIT(CTH_LOG_ERR, "FORCE_OP_PARAM_EXIST fails");
  }
}

CTorchTensor *_get_tensor_by_name(
    Array(CTorchTensor) * tensor_array, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(tensor_array);
  CTorchTensor *target_tensor = NULL;
  for (array_index_t i = 0; i < tensor_array->size; i++) {
    CTorchTensor *tensor = array_at(CTorchTensor)(tensor_array, i);
    if (strcmp(tensor->meta_info->tensor_name, name) == 0) {
      target_tensor = tensor;
      break;
    }
  }

  if (target_tensor == NULL && fail_exit) {
    FAIL_EXIT(CTH_LOG_ERR, "Could not find tensor %s", name);
  } else {
    return target_tensor;
  }
}

CTorchTensor *
get_input_by_name(CTorchOperator *op, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(op);
  return _get_tensor_by_name(op->in_bound_tensors, name, true);
}

CTorchTensor *
get_output_by_name(CTorchOperator *op, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(op);
  return _get_tensor_by_name(op->out_bound_tensors, name, true);
}

void struct_deep_free(CTorchOperator)(CTorchOperator *op) {
  // TODO: implement with array

  // free_list_deep(CTorchTensor)(op->in_bound_tensors);
  // free_list_deep(CTorchTensor)(op->out_bound_tensors);
  // FREE(op);
}