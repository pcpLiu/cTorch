#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_ConstantPad1d_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {

  CTHDim2 *padding;
  cth_extract_param_value(
      op, CTH_PARAM_TYPE_PADDING_D2, (void **)&padding, true);
  cth_tensor_dim_t padding_left = padding->d_0;
  cth_tensor_dim_t padding_right = padding->d_1;

  cth_float_param_t *padding_value_float;
  cth_extract_param_value(
      op,
      CTH_PARAM_TYPE_PADDING_VALUE_FLOAT,
      (void **)&padding_value_float,
      true);

  float padding_fill_val = *padding_value_float;
  CTH_TENSOR_DATA_TYPE data_type =
      cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->meta_info->data_type;
  if (data_type == CTH_TENSOR_DATA_TYPE_INT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_INT_32 ||
      data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    padding_fill_val = (float)((int)padding_fill_val);
  }

  auto m = torch::nn::ConstantPad1d(torch::nn::ConstantPad1dOptions(
      {padding_left, padding_right}, padding_fill_val));
  return m(pytorch_in_tensor);
}

void test_constant_pad_1d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  CTHNode *op_node = create_dummy_op_node_unary_1d_constant_padding(
      CTH_OP_ID_ConstantPad1d, data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_ConstantPad1d_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_nn_op_pytorch(
        op, int16_t, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, int32_t, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad1d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, int64_t, EXPECT_EQ_PRECISION, 1e-3, _ConstantPad1d_pytorch);
  }
}

TEST(cTorchConstantPad1dOpTest, testFloat16Default) {
  test_constant_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchConstantPad1dOpTest, testFloat32Default) {
  test_constant_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchConstantPad1dOpTest, testFloat64Default) {
  test_constant_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchConstantPad1dOpTest, testInt16Default) {
  test_constant_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchConstantPad1dOpTest, testInt32Default) {
  test_constant_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchConstantPad1dOpTest, testInt64Default) {
  test_constant_pad_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}
