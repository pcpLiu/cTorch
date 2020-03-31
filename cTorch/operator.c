#include "cTorch/operator.h"

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
      FAIL_EXIT(CTH_LOG_STR, "Operator does not support data type.");
    }
    tensor_it = tensor_it->next_item;
  }
}