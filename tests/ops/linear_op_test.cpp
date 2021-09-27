#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_Linear_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {

  CTHTensor *weights = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  cth_tensor_dim_t out_feature_dim = weights->meta_info->dims[0];
  cth_tensor_dim_t in_feature_dim = weights->meta_info->dims[1];

  CTHTensor *bias = cth_array_at(CTHTensor)(op->in_bound_tensors, 2);

  auto m = torch::nn::Linear(
      torch::nn::LinearOptions(in_feature_dim, out_feature_dim).bias(true));

  torch::NoGradGuard no_grad;

  // assign weight
  auto weight_py_tensor = create_torch_tensor(weights);
  m->weight.copy_(weight_py_tensor);

  // copy bias
  auto bias_py_tensor = create_torch_tensor(bias);
  m->bias.copy_(bias_py_tensor);

  auto ret = m(pytorch_in_tensor);
  return std::move(ret);
}

void test_linear_1d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  CTHNode *op_node =
      create_dummy_op_node_linear(CTH_OP_ID_Linear, 4, data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_Linear_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _Linear_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _Linear_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_nn_op_pytorch(
        op, int16_t, EXPECT_EQ_PRECISION, 1e-3, _Linear_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, int32_t, EXPECT_EQ_PRECISION, 1e-3, _Linear_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, int64_t, EXPECT_EQ_PRECISION, 1e-3, _Linear_pytorch);
  }
}

TEST(cTorchLinearOpTest, testFloat32Default) {
  test_linear_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
}

TEST(cTorchLinearOpTest, testFloat64Default) {
  test_linear_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
}

TEST(cTorchLinearOpTest, testInt16Default) {
  test_linear_1d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0, 10.0);
}

TEST(cTorchLinearOpTest, testInt32Default) {
  test_linear_1d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -10.0, 10.0);
}

TEST(cTorchLinearOpTest, testInt64Default) {
  test_linear_1d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -10.0, 10.0);
}

TEST(cTorchLinearOpTest, testUInt8Default) {
  test_linear_1d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, -10.0, 10.0);
}
