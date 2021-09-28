#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_Hardshrink_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {
  float *lambda;
  cth_extract_param_value(
      op, CTH_PARAM_TYPE_LAMBD_FLOAT, (void **)&lambda, true);

  auto m = torch::nn::Hardshrink(
      torch::nn::HardshrinkOptions().lambda(PTR_VAL(lambda)));

  auto ret = m(pytorch_in_tensor);
  return std::move(ret);
}

void test_hardshrink(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = 2;
  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_Hardshrink, 1, 1, 1);

  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_LAMBD_FLOAT;
  float alpha = _rand_float(-1.0, 1.0);
  param->data.float_val = &alpha;
  cth_array_set(CTHParam)(op->params, 0, param);

  if (backend == CTH_BACKEND_DEFAULT) {
    op_Hardshrink_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _Hardshrink_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _Hardshrink_pytorch);
  }
}

TEST(cTorchHardshrinkOpTest, testFloat32Default) {
  test_hardshrink(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
}

TEST(cTorchHardshrinkOpTest, testFloat64Default) {
  test_hardshrink(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
}
