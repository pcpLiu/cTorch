#include <iostream>
#include <time.h>

#include "tests/test_util.h"

float _rand_float(float min, float max) {
  return ((float)rand() / (float)RAND_MAX) * (max - min) + min;
}

/**
 * @brief min <= ret <= max
 *
 * @param min
 * @param max
 * @return int
 */
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
    for (cth_tensor_dim_t i = 0; i < tensor->meta_info->n_elements; i++) {     \
      val[i] = (type)rand_func(__VA_ARGS__);                                   \
    }                                                                          \
  } while (0)

#define _print_out_value(type, print_format, in_ptr, out_ptr, i)               \
  do {                                                                         \
    CTH_LOG(                                                                   \
        CTH_LOG_INFO,                                                          \
        print_format,                                                          \
        ((type *)in_ptr)[i],                                                   \
        ((type *)out_ptr)[i]);                                                 \
  } while (0)

#define _print_out_value_triple(                                               \
    type, print_format, in_ptr_1, in_ptr_2, out_ptr, i)                        \
  do {                                                                         \
    CTH_LOG(                                                                   \
        CTH_LOG_INFO,                                                          \
        print_format,                                                          \
        ((type *)in_ptr_1)[i],                                                 \
        ((type *)in_ptr_2)[i],                                                 \
        ((type *)out_ptr)[i]);                                                 \
  } while (0)

CTHTensor *create_dummy_tensor(
    cth_tensor_dim_t *dims,
    cth_tensor_dim_t n_dim,
    CTH_TENSOR_DATA_TYPE data_type,
    float min,
    float max) {
  CTHTensor *tensor = (CTHTensor *)MALLOC_NAME(sizeof(CTHTensor), "tensor");
  tensor->meta_info =
      (CTHTensorMeta *)MALLOC_NAME(sizeof(CTHTensorMeta), "meta info");
  tensor->meta_info->dims = dims;
  tensor->meta_info->n_dim = n_dim;
  tensor->meta_info->data_type = data_type;
  tensor->meta_info->is_sharded = false;
  tensor->meta_info->device = CTH_TENSOR_DEVICE_NORMAL;
  cth_tensor_set_name(tensor, "Dummy Tensor");

  cth_tensor_dim_t n_ele = 1;
  for (cth_tensor_dim_t i = 0; i < tensor->meta_info->n_dim; i++) {
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

CTHNode *create_dummy_op_node_unary(
    CTH_OP_ID op_id,
    cth_tensor_dim_t *dims,
    cth_tensor_dim_t n_dim,
    CTH_TENSOR_DATA_TYPE data_type,
    float min,
    float max) {
  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(1);
  op->out_bound_tensors = cth_new_array(CTHTensor)(1);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

CTHNode *create_dummy_op_node_unary_1d_padding(
    CTH_OP_ID op_id, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t input_n_dim = 3, min_dim = 2, max_dim = 100;
  cth_tensor_dim_t *input_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  _rand_dims(input_dims, input_n_dim, min_dim, max_dim);

  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(1);
  op->out_bound_tensors = cth_new_array(CTHTensor)(1);
  op->params = cth_new_array(CTHParam)(1);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(input_dims, input_n_dim, data_type, min, max));

  // random get padding size
  cth_tensor_dim_t padding_left = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t padding_right = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t *output_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[2] + padding_left + padding_right;

  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(output_dims, input_n_dim, data_type, min, max));

  // padding param
  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_PADDING_D2;
  param->data.dim_2_val = (CTHDim2 *)MALLOC(sizeof(CTHDim2));
  param->data.dim_2_val->d_0 = padding_left;
  param->data.dim_2_val->d_1 = padding_right;
  cth_array_set(CTHParam)(op->params, 0, param);

  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

CTHNode *create_dummy_op_node_unary_1d_constant_padding(
    CTH_OP_ID op_id, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t input_n_dim = 3, min_dim = 2, max_dim = 100;
  cth_tensor_dim_t *input_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  _rand_dims(input_dims, input_n_dim, min_dim, max_dim);

  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(1);
  op->out_bound_tensors = cth_new_array(CTHTensor)(1);
  op->params = cth_new_array(CTHParam)(2);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(input_dims, input_n_dim, data_type, min, max));

  // random get padding size
  cth_tensor_dim_t padding_left = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t padding_right = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t *output_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[2] + padding_left + padding_right;

  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(output_dims, input_n_dim, data_type, min, max));

  // padding param
  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_PADDING_D2;
  param->data.dim_2_val = (CTHDim2 *)MALLOC(sizeof(CTHDim2));
  param->data.dim_2_val->d_0 = padding_left;
  param->data.dim_2_val->d_1 = padding_right;
  cth_array_set(CTHParam)(op->params, 0, param);

  // padding value param
  CTHParam *param1 = (CTHParam *)MALLOC(sizeof(CTHParam));
  param1->type = CTH_PARAM_TYPE_PADDING_VALUE_FLOAT;
  param1->data.float_val = (float *)MALLOC(sizeof(float) * 1);
  *(param1->data.float_val) = _rand_float(min, max);
  cth_array_set(CTHParam)(op->params, 1, param1);

  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

CTHNode *create_dummy_op_node_unary_2d_padding(
    CTH_OP_ID op_id, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t input_n_dim = 4, min_dim = 2, max_dim = 20;
  cth_tensor_dim_t *input_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  _rand_dims(input_dims, input_n_dim, min_dim, max_dim);

  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(1);
  op->out_bound_tensors = cth_new_array(CTHTensor)(1);
  op->params = cth_new_array(CTHParam)(1);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(input_dims, input_n_dim, data_type, min, max));

  // random get padding size
  cth_tensor_dim_t padding_left = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_right = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_top = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t padding_bottom = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t *output_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[2] + padding_top + padding_bottom;
  output_dims[3] = input_dims[3] + padding_left + padding_right;

  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(output_dims, input_n_dim, data_type, min, max));

  // padding param
  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_PADDING_D4;
  param->data.dim_4_val = (CTHDim4 *)MALLOC(sizeof(CTHDim4));
  param->data.dim_4_val->d_0 = padding_left;
  param->data.dim_4_val->d_1 = padding_right;
  param->data.dim_4_val->d_2 = padding_top;
  param->data.dim_4_val->d_3 = padding_bottom;
  cth_array_set(CTHParam)(op->params, 0, param);

  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

CTHNode *create_dummy_op_node_unary_2d_constant_padding(
    CTH_OP_ID op_id, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t input_n_dim = 4, min_dim = 2, max_dim = 20;
  cth_tensor_dim_t *input_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  _rand_dims(input_dims, input_n_dim, min_dim, max_dim);

  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(1);
  op->out_bound_tensors = cth_new_array(CTHTensor)(1);
  op->params = cth_new_array(CTHParam)(2);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(input_dims, input_n_dim, data_type, min, max));

  // random get padding size
  cth_tensor_dim_t padding_left = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_right = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_top = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t padding_bottom = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t *output_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[2] + padding_top + padding_bottom;
  output_dims[3] = input_dims[3] + padding_left + padding_right;

  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(output_dims, input_n_dim, data_type, min, max));

  // padding param
  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_PADDING_D4;
  param->data.dim_4_val = (CTHDim4 *)MALLOC(sizeof(CTHDim4));
  param->data.dim_4_val->d_0 = padding_left;
  param->data.dim_4_val->d_1 = padding_right;
  param->data.dim_4_val->d_2 = padding_top;
  param->data.dim_4_val->d_3 = padding_bottom;
  cth_array_set(CTHParam)(op->params, 0, param);

  // padding value param
  CTHParam *param1 = (CTHParam *)MALLOC(sizeof(CTHParam));
  param1->type = CTH_PARAM_TYPE_PADDING_VALUE_FLOAT;
  param1->data.float_val = (float *)MALLOC(sizeof(float) * 1);
  *(param1->data.float_val) = _rand_float(min, max);
  cth_array_set(CTHParam)(op->params, 1, param1);

  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

CTHNode *create_dummy_op_node_unary_3d_padding(
    CTH_OP_ID op_id, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t input_n_dim = 5, min_dim = 1, max_dim = 10;
  cth_tensor_dim_t *input_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  _rand_dims(input_dims, input_n_dim, min_dim, max_dim);

  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(1);
  op->out_bound_tensors = cth_new_array(CTHTensor)(1);
  op->params = cth_new_array(CTHParam)(1);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(input_dims, input_n_dim, data_type, min, max));

  // random get padding size
  cth_tensor_dim_t padding_left = _rand_int(0, input_dims[4] - 1);
  cth_tensor_dim_t padding_right = _rand_int(0, input_dims[4] - 1);
  cth_tensor_dim_t padding_top = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_bottom = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_front = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t padding_back = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t *output_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[2] + padding_front + padding_back;
  output_dims[3] = input_dims[3] + padding_top + padding_bottom;
  output_dims[4] = input_dims[4] + padding_left + padding_right;

  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(output_dims, input_n_dim, data_type, min, max));

  // padding param
  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_PADDING_D6;
  param->data.dim_6_val = (CTHDim6 *)MALLOC(sizeof(CTHDim6));
  param->data.dim_6_val->d_0 = padding_left;
  param->data.dim_6_val->d_1 = padding_right;
  param->data.dim_6_val->d_2 = padding_top;
  param->data.dim_6_val->d_3 = padding_bottom;
  param->data.dim_6_val->d_4 = padding_front;
  param->data.dim_6_val->d_5 = padding_back;
  cth_array_set(CTHParam)(op->params, 0, param);

  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

CTHNode *create_dummy_op_node_unary_3d_constant_padding(
    CTH_OP_ID op_id, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t input_n_dim = 5, min_dim = 1, max_dim = 10;
  cth_tensor_dim_t *input_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  _rand_dims(input_dims, input_n_dim, min_dim, max_dim);

  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(1);
  op->out_bound_tensors = cth_new_array(CTHTensor)(1);
  op->params = cth_new_array(CTHParam)(2);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(input_dims, input_n_dim, data_type, min, max));

  // random get padding size
  cth_tensor_dim_t padding_left = _rand_int(0, input_dims[4] - 1);
  cth_tensor_dim_t padding_right = _rand_int(0, input_dims[4] - 1);
  cth_tensor_dim_t padding_top = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_bottom = _rand_int(0, input_dims[3] - 1);
  cth_tensor_dim_t padding_front = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t padding_back = _rand_int(0, input_dims[2] - 1);
  cth_tensor_dim_t *output_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * input_n_dim);
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[2] + padding_front + padding_back;
  output_dims[3] = input_dims[3] + padding_top + padding_bottom;
  output_dims[4] = input_dims[4] + padding_left + padding_right;

  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(output_dims, input_n_dim, data_type, min, max));

  // padding param
  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_PADDING_D6;
  param->data.dim_6_val = (CTHDim6 *)MALLOC(sizeof(CTHDim6));
  param->data.dim_6_val->d_0 = padding_left;
  param->data.dim_6_val->d_1 = padding_right;
  param->data.dim_6_val->d_2 = padding_top;
  param->data.dim_6_val->d_3 = padding_bottom;
  param->data.dim_6_val->d_4 = padding_front;
  param->data.dim_6_val->d_5 = padding_back;
  cth_array_set(CTHParam)(op->params, 0, param);

  // padding value param
  CTHParam *param1 = (CTHParam *)MALLOC(sizeof(CTHParam));
  param1->type = CTH_PARAM_TYPE_PADDING_VALUE_FLOAT;
  param1->data.float_val = (float *)MALLOC(sizeof(float) * 1);
  *(param1->data.float_val) = _rand_float(min, max);
  cth_array_set(CTHParam)(op->params, 1, param1);

  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->conent.op = op;
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  return node;
}

CTHOperator *create_dummy_op(
    CTH_OP_ID op_id,
    cth_array_index_t num_inputs,
    cth_array_index_t num_outputs) {
  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(num_inputs);
  op->out_bound_tensors = cth_new_array(CTHTensor)(num_outputs);
  op->params = cth_new_array(CTHParam)(0);
  return op;
}

CTHOperator *create_dummy_op_with_param(
    CTH_OP_ID op_id,
    cth_array_index_t num_inputs,
    cth_array_index_t num_outputs,
    cth_array_index_t num_param) {
  CTHOperator *op = (CTHOperator *)MALLOC(sizeof(CTHOperator));
  op->op_id = op_id;
  op->in_bound_tensors = cth_new_array(CTHTensor)(num_inputs);
  op->out_bound_tensors = cth_new_array(CTHTensor)(num_outputs);
  op->params = cth_new_array(CTHParam)(num_param);
  return op;
}

CTHGraph *create_dummy_graph(cth_array_index_t num_nodes) {
  CTHGraph *graph = (CTHGraph *)MALLOC(sizeof(CTHGraph));
  graph->node_list = cth_new_array(CTHNode)(num_nodes);
  return graph;
}

CTHNode *create_dummy_node(
    node_id_t id,
    cth_array_index_t inbound_size,
    cth_array_index_t outbound_size) {
  CTHNode *node = (CTHNode *)MALLOC(sizeof(CTHNode));
  node->node_type = CTH_NODE_TYPE_OPERATOR;
  node->node_id = id;
  node->inbound_nodes = cth_new_array(CTHNode)(inbound_size);
  node->outbound_nodes = cth_new_array(CTHNode)(outbound_size);
  return node;
}

void sample_print(
    CTH_TENSOR_DATA_TYPE data_type,
    void *in_ptr,
    void *out_ptr,
    cth_tensor_dim_t i) {
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

void sample_print_triple(
    CTH_TENSOR_DATA_TYPE data_type,
    void *in_ptr_1,
    void *in_ptr_2,
    void *out_ptr,
    cth_tensor_dim_t i) {
  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _print_out_value_triple(
        float,
        "input: %f, input: %f, output: %f",
        in_ptr_1,
        in_ptr_2,
        out_ptr,
        i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _print_out_value_triple(
        double,
        "input: %f, input: %f, output: %f",
        in_ptr_1,
        in_ptr_2,
        out_ptr,
        i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _print_out_value_triple(
        int16_t,
        "input: %d, input: %d, output: %d",
        in_ptr_1,
        in_ptr_2,
        out_ptr,
        i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _print_out_value_triple(
        int32_t,
        "input: %d, input: %d, output: %d",
        in_ptr_1,
        in_ptr_2,
        out_ptr,
        i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _print_out_value_triple(
        int64_t,
        "input: %ld, input: %ld, output: %ld",
        in_ptr_1,
        in_ptr_2,
        out_ptr,
        i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _print_out_value_triple(
        uint8_t,
        "input: %u, input: %u, output: %u",
        in_ptr_1,
        in_ptr_2,
        out_ptr,
        i);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _print_out_value_triple(
        bool,
        "input: %d, input: %d, output: %d",
        in_ptr_1,
        in_ptr_2,
        out_ptr,
        i);
  }
}

void _rand_dims(
    cth_tensor_dim_t *dims,
    cth_tensor_dim_t n_dim,
    cth_tensor_dim_t min,
    cth_tensor_dim_t max) {
  for (cth_tensor_dim_t i = 0; i < n_dim; i++) {
    dims[i] = _rand_int(min, max);
  }
}

void _get_reduce_dims(
    cth_tensor_dim_t *dims,
    cth_tensor_dim_t n_dim,
    cth_tensor_dim_t reduce_dim,
    cth_tensor_dim_t *reduce_dims) {
  for (cth_tensor_dim_t i = 0, j = 0; i < n_dim; i++) {
    if (i == reduce_dim) {
      continue;
    }

    reduce_dims[j] = dims[i];
    j++;
  }
}

void _print_index(cth_tensor_dim_t *dims, cth_tensor_dim_t n_dim) {
  printf("dims: [");
  for (cth_tensor_dim_t i = 0; i < n_dim; i++) {
    printf("%ld ", dims[i]);
  }
  printf("]\n");
}
