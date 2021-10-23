#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_matmul_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {
  CTHTensor *in_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  auto pytorch_in_tensor_1 = create_torch_tensor(in_1);

  CTHTensor *in_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  auto pytorch_in_tensor_2 = create_torch_tensor(in_2);

  auto ret = at::matmul(pytorch_in_tensor_1, pytorch_in_tensor_2);

  return std::move(ret);
}

#define _matmul_test_logic()                                                   \
  do {                                                                         \
    if (backend == CTH_BACKEND_DEFAULT) {                                      \
      op_matmul_cpu(op);                                                       \
    }                                                                          \
                                                                               \
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||                          \
        data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                          \
      _ele_wise_equal_nn_op_pytorch(                                           \
          op, float, EXPECT_EQ_PRECISION, 1e-3, _matmul_pytorch);              \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      _ele_wise_equal_nn_op_pytorch(                                           \
          op, double, EXPECT_EQ_PRECISION, 1e-3, _matmul_pytorch);             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      _ele_wise_equal_nn_op_pytorch(                                           \
          op, int16_t, EXPECT_EQ_PRECISION, 1e-3, _matmul_pytorch);            \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      _ele_wise_equal_nn_op_pytorch(                                           \
          op, int32_t, EXPECT_EQ_PRECISION, 1e-3, _matmul_pytorch);            \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      _ele_wise_equal_nn_op_pytorch(                                           \
          op, int64_t, EXPECT_EQ_PRECISION, 1e-3, _matmul_pytorch);            \
    }                                                                          \
  } while (0)

void test_matmul_2d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_matmul, 2, 1, 1);

  cth_tensor_dim_t *dims, *dims2, *dims3;
  cth_tensor_dim_t n_dim, n_dim_2, n_dim_3;

  /////////////////////////////////////////
  // 1) vec X vec

  dims = (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * 1);
  dims[0] = _rand_int(1, 100);
  n_dim = 1;
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  dims2 = (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * 1);
  dims2[0] = dims[0];
  n_dim_2 = 1;
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      1,
      create_dummy_tensor(dims2, n_dim_2, data_type, min, max));

  dims3 = (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * 1);
  dims3[0] = 1;
  n_dim_3 = 1;
  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(dims3, n_dim_3, data_type, min, max));

  _matmul_test_logic();

  /////////////////////////////////////////
  // 2) Matrix X Matrix

  dims = (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * 2);
  // TODO: libtorch will raise segment fault if too many elements.
  // Maybe a bug in libtorch.
  dims[0] = (cth_tensor_dim_t)_rand_int(1, 10);
  dims[1] = (cth_tensor_dim_t)_rand_int(1, 10);
  n_dim = 2;
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  dims2 = (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * 2);
  dims2[0] = dims[1];
  dims2[1] = (cth_tensor_dim_t)_rand_int(1, 10);
  n_dim_2 = 2;
  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      1,
      create_dummy_tensor(dims2, n_dim_2, data_type, min, max));

  dims3 = (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * 2);
  dims3[0] = dims[0];
  dims3[1] = dims2[1];
  n_dim_3 = 2;
  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(dims3, n_dim_3, data_type, min, max));

  _matmul_test_logic();
}

TEST(cTorchMatmulOpTest, testFloat16Default) {
  test_matmul_2d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -1.0, 1.0);
}

TEST(cTorchMatmulOpTest, testFloat32Default) {
  test_matmul_2d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchMatmulOpTest, testFloat64Default) {
  test_matmul_2d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchMatmulOpTest, testInt16Default) {
  test_matmul_2d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0, 10.0);
}

TEST(cTorchMatmulOpTest, testInt32Default) {
  test_matmul_2d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -10.0, 10.0);
}

TEST(cTorchMatmulOpTest, testInt64Default) {
  test_matmul_2d(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -10.0, 10.0);
}
