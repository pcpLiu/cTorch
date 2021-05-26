#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_ConstantPad3d_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {

  CTHDim6 *padding;
  cth_extract_param_value(
      op, CTH_PARAM_TYPE_PADDING_D6, (void **)&padding, true);
  cth_tensor_dim_t padding_left = padding->d_0;
  cth_tensor_dim_t padding_right = padding->d_1;
  cth_tensor_dim_t padding_top = padding->d_2;
  cth_tensor_dim_t padding_bottom = padding->d_3;
  cth_tensor_dim_t padding_front = padding->d_4;
  cth_tensor_dim_t padding_back = padding->d_5;

  cth_float_param_t *padding_value_float;
  cth_extract_param_value(
      op,
      CTH_PARAM_TYPE_PADDING_VALUE_FLOAT,
      (void **)&padding_value_float,
      true);

  // cth_pad_t *padding;
  // EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_PADDING_D6, padding, padding);
  // cth_tensor_dim_t padding_left = padding[0];
  // cth_tensor_dim_t padding_right = padding[1];
  // cth_tensor_dim_t padding_top = padding[2];
  // cth_tensor_dim_t padding_bottom = padding[3];
  // cth_tensor_dim_t padding_front = padding[4];
  // cth_tensor_dim_t padding_back = padding[5];

  // cth_float_param_t *padding_value_float;
  // EXTRACT_PARAM_VALUE(
  //     op,
  //     CTH_PARAM_TYPE_PADDING_VALUE_FLOAT,
  //     padding_value_float,
  //     padding_value_float);

  float padding_fill_val = *padding_value_float;
  CTH_TENSOR_DATA_TYPE data_type =
      cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->meta_info->data_type;
  if (data_type == CTH_TENSOR_DATA_TYPE_INT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_INT_32 ||
      data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    padding_fill_val = floor(padding_fill_val);
  }

  auto m = torch::nn::ConstantPad3d(torch::nn::ConstantPad3dOptions(
      {padding_left,
       padding_right,
       padding_top,
       padding_bottom,
       padding_front,
       padding_back},
      padding_fill_val));
  return m(pytorch_in_tensor);
}

void test_constant_pad_3d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  CTHNode *op_node = create_dummy_op_node_unary_3d_constant_padding(
      CTH_OP_ID_ConstantPad3d, data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_ConstantPad3d_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_nn_op_pytorch(
        op, int16_t, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, int32_t, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, int64_t, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad3d_pytorch);
  }
}

TEST(cTorchConstantPad3dOpTest, testFloat16Default) {
  test_constant_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchConstantPad3dOpTest, testFloat32Default) {
  test_constant_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchConstantPad3dOpTest, testFloat64Default) {
  test_constant_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchConstantPad3dOpTest, testInt16Default) {
  test_constant_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchConstantPad3dOpTest, testInt32Default) {
  test_constant_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchConstantPad3dOpTest, testInt64Default) {
  test_constant_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}
