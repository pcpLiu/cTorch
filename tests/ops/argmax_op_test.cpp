#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor __argmax_pytorch(torch::Tensor &pytorch_in_tensor,
                               cth_tensor_dim_t reduce_dim) {
  return pytorch_in_tensor.argmax(reduce_dim, false);
}

void test_argmax(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                 float max) {
  cth_tensor_dim_t n_dim = 5, min_dim = 1, max_dim = 10;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * n_dim);
  _rand_dims(dims, n_dim, min_dim, max_dim);

  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_argmax, 1, 1, 1);

  CTHTensor *input = create_dummy_tensor(dims, n_dim, data_type, min, max);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);

  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->data.dim = _rand_int(0, n_dim - 1);
  param->type = CTH_PARAM_TYPE_DIM_INT32;
  cth_array_set(CTHParam)(op->params, 0, param);

  cth_tensor_dim_t n_dim_out = n_dim - 1;
  cth_tensor_dim_t *out_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * n_dim_out);
  _get_reduce_dims(dims, n_dim, param->data.dim, out_dims);
  CTHTensor *output = create_dummy_tensor(
      out_dims, n_dim_out, CTH_TENSOR_DATA_TYPE_INT_64, min, max);
  cth_array_set(CTHTensor)(op->out_bound_tensors, 0, output);

  if (backend == CTH_BACKEND_DEFAULT) {
    op_argmax_cpu(op);
  }

  _reduce_typing_test_flow(op, data_type, CTH_TENSOR_DATA_TYPE_INT_64, int64_t,
                           __argmax_pytorch, EXPECT_EQ);
}

TEST(cTorchArgmaxOpTest, testFloat16Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -100.0,
              100.0);
}

TEST(cTorchArgmaxOpTest, testFloat32Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0,
              100.0);
}

TEST(cTorchArgmaxOpTest, testFloat64Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0,
              100.0);
}

TEST(cTorchArgmaxOpTest, testInt16Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0, 10.0);
}

TEST(cTorchArgmaxOpTest, testInt32Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -100.0, 100.0);
}

TEST(cTorchArgmaxOpTest, testInt64Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -100.0, 100.0);
}

TEST(cTorchArgmaxOpTest, testUInt8Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0, 10.0);
}
