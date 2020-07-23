#include <time.h>

#include "tests/test_util.h"

float _rand_float(float min, float max) {
  return ((float)rand() / (float)RAND_MAX) * (max - min) + min;
}

int _rand_int(int min, int max) { return (rand() % (max - min + 1)) + min; }

bool _rand_bool(void) {
  if (_rand_int(0, 1) > 0.5) {
    return false;
  } else {
    return true;
  }
}

#define _fill_tensor(type, n_ele, tensor, rand_func, ...)                      \
  do {                                                                         \
    tensor->meta_info->value_size_of = sizeof(type);                           \
    tensor->values =                                                           \
        (type *)MALLOC_NAME(sizeof(type) * n_ele, "tensor values");            \
    type *val = (type *)tensor->values;                                        \
    for (tensor_size_t i = 0; i < tensor->meta_info->n_elements; i++) {        \
      val[i] = (type)rand_func(__VA_ARGS__);                                   \
    }                                                                          \
  } while (0)

#define _print_out_value(type, print_format, in_ptr, out_ptr, i)               \
  do {                                                                         \
    CTH_LOG(CTH_LOG_INFO, print_format, ((type *)in_ptr)[i],                   \
            ((type *)out_ptr)[i]);                                             \
  } while (0)

#define _print_out_value_triple(type, print_format, in_ptr_1, in_ptr_2,        \
                                out_ptr, i)                                    \
  do {                                                                         \
    CTH_LOG(CTH_LOG_INFO, print_format, ((type *)in_ptr_1)[i],                 \
            ((type *)in_ptr_2)[i], ((type *)out_ptr)[i]);                      \
  } while (0)

CTorchTensor *create_dummy_tensor(tensor_dim_t *dims, tensor_dim_t n_dim,
                                  CTH_TENSOR_DATA_TYPE data_type, float min,
                                  float max) {
  CTorchTensor *tensor =
      (CTorchTensor *)MALLOC_NAME(sizeof(CTorchTensor), "tensor");
  tensor->meta_info =
      (CTorchTensorMeta *)MALLOC_NAME(sizeof(CTorchTensorMeta), "meta info");
  tensor->meta_info->dims = dims;
  tensor->meta_info->n_dim = n_dim;
  tensor->meta_info->data_type = data_type;
  tensor->meta_info->is_sharded = false;
  tensor->meta_info->device = CTH_TENSOR_DEVICE_NORMAL;
  cth_tensor_set_name(tensor, "Dummy Tensor");

  tensor_size_t n_ele = 1;
  for (tensor_size_t i = 0; i < tensor->meta_info->n_dim; i++) {
    n_ele *= tensor->meta_info->dims[i];
  }
  tensor->meta_info->n_elements = n_ele;
  tensor->meta_info->data_type = data_type;

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _fill_tensor(float, n_ele, tensor, _rand_float, (float)min, (float)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _fill_tensor(double, n_ele, tensor, _rand_float, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _fill_tensor(int16_t, n_ele, tensor, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _fill_tensor(int32_t, n_ele, tensor, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _fill_tensor(int64_t, n_ele, tensor, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _fill_tensor(uint8_t, n_ele, tensor, _rand_int, (int)min, (int)max);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _fill_tensor(bool, n_ele, tensor, _rand_bool);
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

void sample_print(CTH_TENSOR_DATA_TYPE data_type, void *in_ptr, void *out_ptr,
                  tensor_size_t i) {
  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _print_out_value(float, "input: %f, output: %f", in_ptr, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _print_out_value(double, "input: %f, output: %f", in_ptr, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _print_out_value(int16_t, "input: %d, output: %d", in_ptr, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _print_out_value(int32_t, "input: %d, output: %d", in_ptr, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _print_out_value(int64_t, "input: %ld, output: %ld", in_ptr, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _print_out_value(uint8_t, "input: %u, output: %u", in_ptr, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _print_out_value(bool, "input: %d, output: %d", in_ptr, out_ptr, i);
  }
}

void sample_print_triple(CTH_TENSOR_DATA_TYPE data_type, void *in_ptr_1,
                         void *in_ptr_2, void *out_ptr, tensor_size_t i) {
  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _print_out_value_triple(float, "input: %f, input: %f, output: %f", in_ptr_1,
                            in_ptr_2, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _print_out_value_triple(double, "input: %f, input: %f, output: %f",
                            in_ptr_1, in_ptr_2, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _print_out_value_triple(int16_t, "input: %d, input: %d, output: %d",
                            in_ptr_1, in_ptr_2, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _print_out_value_triple(int32_t, "input: %d, input: %d, output: %d",
                            in_ptr_1, in_ptr_2, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _print_out_value_triple(int64_t, "input: %ld, input: %ld, output: %ld",
                            in_ptr_1, in_ptr_2, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _print_out_value_triple(uint8_t, "input: %u, input: %u, output: %u",
                            in_ptr_1, in_ptr_2, out_ptr, i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _print_out_value_triple(bool, "input: %d, input: %d, output: %d", in_ptr_1,
                            in_ptr_2, out_ptr, i);
  }
}

void _rand_dims(tensor_dim_t *dims, tensor_dim_t n_dim, tensor_dim_t min,
                tensor_dim_t max) {
  for (tensor_dim_t i = 0; i < n_dim; i++) {
    dims[i] = _rand_int(min, max);
  }
}

void _get_reduce_dims(tensor_dim_t *dims, tensor_dim_t n_dim,
                      tensor_dim_t reduce_dim, tensor_dim_t *reduce_dims) {
  for (tensor_dim_t i = 0, j = 0; i < n_dim; i++) {
    if (i == reduce_dim) {
      continue;
    }
    reduce_dims[j] = dims[i];
    j++;
  }
}

void _print_index(tensor_dim_t *dims, tensor_dim_t n_dim) {
  printf("dims: [");
  for (tensor_dim_t i = 0; i < n_dim; i++) {
    printf("%d ", dims[i]);
  }
  printf("]\n");
}
