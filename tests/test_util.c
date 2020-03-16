#include "cTorch/c_torch.h"
#include "cTorch/operators/x86/op_list.h"
#include "stdlib.h"

float rand_float(float min, float max) {
  return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
}

CTorchTensor *create_dummy_tensor(uint32_t *dims) {
  CTorchTensor *tensor = (CTorchTensor *)MALLOC(sizeof(CTorchTensor));
  tensor->meta_info = (CTorchTensorMeta *)MALLOC(sizeof(CTorchTensorMeta));
  tensor->meta_info->dim_size_list = dims;
  tensor->meta_info->n_dim = sizeof(dims) / sizeof(dims[0]);
  uint64_t n_ele = 1;
  for (int i = 0; i < tensor->meta_info->n_dim; i++) {
    n_ele *= tensor->meta_info->dim_size_list[i];
  }
  tensor->meta_info->n_elements = n_ele;
  tensor->meta_info->data_type = CTH_TENSOR_DATA_TYPE_FLOAT_32;
  tensor->values = (float *)MALLOC(sizeof(float) * n_ele);
  float *val = (float *)tensor->values;
  for (int i = 0; i < tensor->meta_info->n_elements; i++) {
    val[i] = rand_float(-1.0, 1.0);
  }

  return tensor;
}

CTorchNode *create_dummy_op_node(CTH_OP_ID op_id, uint32_t *dims) {
  CTorchOperator *op = (CTorchOperator *)MALLOC(sizeof(CTorchOperator));
  op->op_id = op_id;
  op->in_bound_tensors = new_list(CTorchTensor)();
  op->out_bound_tensors = new_list(CTorchTensor)();
  insert_list(CTorchTensor)(op->in_bound_tensors, create_dummy_tensor(dims));
  insert_list(CTorchTensor)(op->out_bound_tensors, create_dummy_tensor(dims));

  CTorchNode *node = (CTorchNode *)MALLOC(sizeof(CTorchNode));
  node->conent.op = op;
  node->exe_status = CTH_NODE_EXE_STATUS_CLEAN;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}
