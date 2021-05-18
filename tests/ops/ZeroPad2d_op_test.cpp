#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_ZeroPad2d_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {

  cth_pad_t *padding;
  EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_PADDING_D4, padding, padding);
  cth_tensor_dim_t padding_left = padding[0];
  cth_tensor_dim_t padding_right = padding[1];
  cth_tensor_dim_t padding_top = padding[2];
  cth_tensor_dim_t padding_bottom = padding[3];
  auto m = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(
      {padding_left, padding_right, padding_top, padding_bottom}));
  return m(pytorch_in_tensor);
}

void test_zero_pad_2d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  CTHNode *op_node = create_dummy_op_node_unary_2d_padding(
      CTH_OP_ID_ReplicationPad2d, data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_ZeroPad2d_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _ZeroPad2d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _ZeroPad2d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_nn_op_pytorch(
        op, int16_t, EXPECT_EQ_PRECISION, 1e-3, _ZeroPad2d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, int32_t, EXPECT_EQ_PRECISION, 1e-3, _ZeroPad2d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, int64_t, EXPECT_EQ_PRECISION, 1e-3, _ZeroPad2d_pytorch);
  }
}

TEST(cTorchZeroPad2dOpTest, testFloat16Default) {
  test_zero_pad_2d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchZeroPad2dOpTest, testFloat32Default) {
  test_zero_pad_2d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchZeroPad2dOpTest, testFloat64Default) {
  test_zero_pad_2d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchZeroPad2dOpTest, testInt16Default) {
  test_zero_pad_2d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchZeroPad2dOpTest, testInt32Default) {
  test_zero_pad_2d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchZeroPad2dOpTest, testInt64Default) {
  test_zero_pad_2d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}
