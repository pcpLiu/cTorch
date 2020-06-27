#include <time.h>

#include "tests/test_util.h"

float _rand_float(float min, float max) {
  return ((float)rand() / (float)RAND_MAX) * (max - min) + min;
}

int _rand_int(int min, int max) { return (rand() % (max - min + 1)) + min; }

bool _rand_bool(void) {
  if (_rand_int(0, 1) == 0) {
    return false;
  } else {
    return true;
  }
}

#define _fill_tensor(type, rand_func, ...)                                     \
  {                                                                            \
    tensor->values = (type *)MALLOC(sizeof(type) * n_ele);                     \
    type *val = (type *)tensor->values;                                        \
    for (int i = 0; i < tensor->meta_info->n_elements; i++) {                  \
      val[i] = rand_func(__VA_ARGS__);                                         \
    }                                                                          \
  }

CTorchTensor *create_dummy_tensor(tensor_dim_t *dims, tensor_dim_t n_dim,
                                  CTH_TENSOR_DATA_TYPE data_type, float min,
                                  float max) {
  CTorchTensor *tensor = (CTorchTensor *)MALLOC(sizeof(CTorchTensor));
  tensor->meta_info = (CTorchTensorMeta *)MALLOC(sizeof(CTorchTensorMeta));
  tensor->meta_info->dims = dims;
  tensor->meta_info->n_dim = n_dim;
  tensor->meta_info->data_type = data_type;
  cth_tensor_set_name(tensor, "Dummy Tensor");

  uint64_t n_ele = 1;
  for (int i = 0; i < tensor->meta_info->n_dim; i++) {
    n_ele *= tensor->meta_info->dims[i];
  }
  tensor->meta_info->n_elements = n_ele;
  tensor->meta_info->data_type = data_type;

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _fill_tensor(float, _rand_float, (float)min, (float)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _fill_tensor(double, _rand_float, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _fill_tensor(int16_t, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _fill_tensor(int32_t, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _fill_tensor(int64_t, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _fill_tensor(uint8_t, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _fill_tensor(char, _rand_bool);
  }

  return tensor;
}

CTorchNode *create_dummy_op_node_unary(CTH_OP_ID op_id, tensor_dim_t *dims,
                                       tensor_dim_t n_dim,
                                       CTH_TENSOR_DATA_TYPE data_type,
                                       float min, float max) {
  CTorchOperator *op = (CTorchOperator *)MALLOC(sizeof(CTorchOperator));
  op->op_id = op_id;
  op->in_bound_tensors = new_array(CTorchTensor)(1);
  op->out_bound_tensors = new_array(CTorchTensor)(1);
  array_set(CTorchTensor)(
      op->in_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(
      op->out_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  CTorchNode *node = (CTorchNode *)MALLOC(sizeof(CTorchNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

bool tensor_all_nan(CTorchTensor *tensor) {
  bool ret = true;

  void *val = tensor->values;
  if (tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    for (uint64_t i = 0; i < tensor->meta_info->n_elements; i++) {
      if (((float *)val)[i] == ((float *)val)[i]) {
        ret = false;
        break;
      }
    }
  } else if (tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    for (uint64_t i = 0; i < tensor->meta_info->n_elements; i++) {
      if (((double *)val)[i] == ((double *)val)[i]) {
        ret = false;
        break;
      }
    }
  } else if (tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    for (uint64_t i = 0; i < tensor->meta_info->n_elements; i++) {
      if (((int16_t *)val)[i] == ((int16_t *)val)[i]) {
        ret = false;
        break;
      }
    }
  } else if (tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    for (uint64_t i = 0; i < tensor->meta_info->n_elements; i++) {
      if (((int32_t *)val)[i] == ((int32_t *)val)[i]) {
        ret = false;
        break;
      }
    }
  } else if (tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    for (uint64_t i = 0; i < tensor->meta_info->n_elements; i++) {
      if (((int64_t *)val)[i] == ((int64_t *)val)[i]) {
        ret = false;
        break;
      }
    }
  } else if (tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    for (uint64_t i = 0; i < tensor->meta_info->n_elements; i++) {
      if (((uint8_t *)val)[i] == ((uint8_t *)val)[i]) {
        ret = false;
        break;
      }
    }
  } else if (tensor->meta_info->data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    for (uint64_t i = 0; i < tensor->meta_info->n_elements; i++) {
      if (((bool *)val)[i] == ((bool *)val)[i]) {
        ret = false;
        break;
      }
    }
  }

  return ret;
}

CTorchOperator *create_dummy_op(CTH_OP_ID op_id, array_index_t num_inputs,
                                array_index_t num_outputs) {
  CTorchOperator *op = MALLOC(sizeof(CTorchOperator));
  op->op_id = op_id;
  op->in_bound_tensors = new_array(CTorchTensor)(num_inputs);
  op->out_bound_tensors = new_array(CTorchTensor)(num_outputs);
  op->params = new_array(CTorchParam)(0);
  return op;
}

CTorchOperator *create_dummy_op_with_param(CTH_OP_ID op_id,
                                           array_index_t num_inputs,
                                           array_index_t num_outputs,
                                           array_index_t num_param) {
  CTorchOperator *op = MALLOC(sizeof(CTorchOperator));
  op->op_id = op_id;
  op->in_bound_tensors = new_array(CTorchTensor)(num_inputs);
  op->out_bound_tensors = new_array(CTorchTensor)(num_outputs);
  op->params = new_array(CTorchParam)(num_param);
  return op;
}

CTorchGraph *create_dummy_graph(array_index_t num_nodes) {
  CTorchGraph *graph = MALLOC(sizeof(CTorchGraph));
  graph->node_list = new_array(CTorchNode)(num_nodes);
  return graph;
}

CTorchNode *create_dummy_node(node_id_t id, array_index_t inbound_size,
                              array_index_t outbound_size) {
  CTorchNode *node = (CTorchNode *)MALLOC(sizeof(CTorchNode));
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  node->node_id = id;
  node->inbound_nodes = new_array(CTorchNode)(inbound_size);
  node->outbound_nodes = new_array(CTorchNode)(outbound_size);
  return node;
}
