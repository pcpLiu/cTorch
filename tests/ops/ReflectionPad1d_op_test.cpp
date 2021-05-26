#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_ReflectivePad1d_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {

  CTHDim2 *padding;
  cth_extract_param_value(
      op, CTH_PARAM_TYPE_PADDING_D2, (void **)&padding, true);
  cth_tensor_dim_t padding_left = padding->d_0;
  cth_tensor_dim_t padding_right = padding->d_1;
  auto m = torch::nn::ReflectionPad1d(
      torch::nn::ReflectionPad1dOptions({padding_left, padding_right}));
  return m(pytorch_in_tensor);
}

void test_reflective_pad_1d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  CTHNode *op_node = create_dummy_op_node_unary_1d_padding(
      CTH_OP_ID_ReplicationPad1d, data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_ReflectionPad1d_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _ReflectivePad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _ReflectivePad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_nn_op_pytorch(
        op, int16_t, EXPECT_EQ_PRECISION, 1e-3, _ReflectivePad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, int32_t, EXPECT_EQ_PRECISION, 1e-3, _ReflectivePad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, int64_t, EXPECT_EQ_PRECISION, 1e-3, _ReflectivePad1d_pytorch);
  }
}

TEST(cTorchReflectionPad1dOpTest, testFloat16Default) {
  test_reflective_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchReflectionPad1dOpTest, testFloat32Default) {
  test_reflective_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchReflectionPad1dOpTest, testFloat64Default) {
  test_reflective_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}
