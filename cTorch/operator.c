#include "cTorch/operator.h"
#include "cTorch/consts.h"
#include "cTorch/debug_util.h"
#include "cTorch/logger_util.h"

#include <string.h>

impl_new_list_item_func(CTorchOperator);
impl_new_list_func(CTorchOperator);
impl_insert_list_func(CTorchOperator);
impl_list_at_func(CTorchOperator);
impl_list_pop_func(CTorchOperator);
impl_free_list_func(CTorchOperator);
impl_free_list_deep_func(CTorchOperator);

void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTorchOperator *op) {
  FAIL_NULL_PTR(op);

  if (op->in_bound_tensors->size != op->out_bound_tensors->size) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "Operator should have same numbers of input and output tensors.");
  }
}

void OP_FAIL_ON_DTYPE(CTorchOperator *op, CTH_TENSOR_DATA_TYPE data_type) {
  FAIL_NULL_PTR(op);

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

void FORCE_OP_INPUT_EXIST(
    CTorchOperator *op, const char *name, CTH_TENSOR_DATA_TYPE data_type) {
  FAIL_NULL_PTR(op);

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
    FAIL_EXIT(CTH_LOG_ERR, "FORCE_OP_INPUT_EXIST fails");
  }
}

void FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(
    const CTorchOperator *op,
    const array_index_t num_input,
    const array_index_t num_output) {
  FAIL_NULL_PTR(op);

  if (num_input != op->in_bound_tensors->size) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "FORCE_OP_INPUT_OUTPUT_TENSOR_NUM failS. Required No. of input tensors "
        "is %u, given %u",
        num_input,
        op->in_bound_tensors->size);
  }

  if (num_output != op->out_bound_tensors->size) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "FORCE_OP_INPUT_OUTPUT_TENSOR_NUM fails. Required No. of output "
        "tensors "
        "is %u, given %u",
        num_output,
        op->out_bound_tensors->size);
  }
}

void FORCE_OP_PARAM_NUM(
    const CTorchOperator *op, const array_index_t num_param) {
  FAIL_NULL_PTR(op);

  if (op->params->size != num_param) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "FORCE_OP_PARAM_NUM fails. Required No. of output params is %d, op has "
        "%d",
        num_param,
        op->out_bound_tensors->size);
  }
}

void FORCE_OP_PARAM_EXIST(CTorchOperator *op, const CTH_PARAM_TYPE type) {
  FAIL_NULL_PTR(op);
  for (array_index_t i = 0; i < op->params->size; i++) {
    CTorchParam *param = array_at(CTorchParam)(op->params, i);
    if (type == param->type) {
      return;
    }
  }

  FAIL_EXIT(CTH_LOG_ERR, "FORCE_OP_PARAM_EXIST failed.");
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
cth_get_input_by_name(CTorchOperator *op, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(op);
  return _get_tensor_by_name(op->in_bound_tensors, name, true);
}

CTorchTensor *
get_output_by_name(CTorchOperator *op, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(op);
  return _get_tensor_by_name(op->out_bound_tensors, name, true);
}

void struct_deep_free(CTorchOperator)(CTorchOperator *op) {
  free_array_deep(CTorchTensor)(op->in_bound_tensors);
  free_array_deep(CTorchTensor)(op->out_bound_tensors);
  free_array_deep(CTorchParam)(op->params);
  FREE(op);
}

CTorchParam *cth_get_param_by_type(
    CTorchOperator *op, const CTH_PARAM_TYPE type, bool fail_exit) {
  FAIL_NULL_PTR(op);

  for (int i = 0; i < op->params->size; i++) {
    CTorchParam *param = array_at(CTorchParam)(op->params, i);
    if (type == param->type) {
      return param;
    }
  }

  if (fail_exit) {
    FAIL_EXIT(CTH_LOG_ERR, "Cannot find param with type %u", type);
  }
}
