#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_PReLU_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {
  cth_tensor_dim_t *num_parameters;
  cth_extract_param_value(
      op, CTH_PARAM_TYPE_NUM_PARAMETERS, (void **)&num_parameters, true);

  auto m = torch::nn::PReLU(
      torch::nn::PReLUOptions().num_parameters(PTR_VAL(num_parameters)));

  torch::NoGradGuard no_grad;

  // assign weight
  CTHTensor *weight = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  auto weight_py_tensor = create_torch_tensor(weight);
  m->weight.copy_(weight_py_tensor);

  auto ret = m(pytorch_in_tensor);
  return std::move(ret);
}

void test_prelu_1d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t dims[] = {
      _rand_int(1, 5), _rand_int(1, 10), _rand_int(1, 10)};
  cth_tensor_dim_t n_dim = 3;
  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_add, 2, 1, 1);

  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  cth_tensor_dim_t num_parameters = 1;
  if (_rand_int(1, 5) > 2) {
    num_parameters = dims[1];
  }
  cth_tensor_dim_t weight_dims[] = {num_parameters};
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      1,
      create_dummy_tensor(weight_dims, 1, data_type, min, max));

  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_NUM_PARAMETERS;
  param->data.dim_val = &num_parameters;
  cth_array_set(CTHParam)(op->params, 0, param);

  if (backend == CTH_BACKEND_DEFAULT) {
    op_PReLU_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _PReLU_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _PReLU_pytorch);
  }
}

TEST(cTorchPReLUOpTest, testFloat32Default) {
  test_prelu_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
}

TEST(cTorchPReLUOpTest, testFloat64Default) {
  test_prelu_1d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
}
