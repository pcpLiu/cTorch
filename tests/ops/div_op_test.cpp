#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _cth_test_div(a, b) (a / b)

void test_div(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
              float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = 2;
  CTorchOperator *op = create_dummy_op(CTH_OP_ID_add, 2, 1);
  array_set(CTorchTensor)(
      op->in_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(
      op->in_bound_tensors, 1,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(
      op->out_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  if (backend == CTH_BACKEND_DEFAULT) {
    op_div_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_div_mkl(op);
  } else if (backend == CTH_BACKEND_APPLE) {
#ifdef BACKEND_APPLE
    op_div_apple(op);
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    op_div_cuda(op);
#endif
  }

  sample_print_triple(
      data_type, array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
      array_at(CTorchTensor)(op->in_bound_tensors, 1)->values,
      array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_binary(op, float, EXPECT_FLOAT_EQ, _cth_test_div);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_binary(op, double, EXPECT_DOUBLE_EQ, _cth_test_div);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_binary(op, int16_t, EXPECT_EQ, _cth_test_div);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_binary(op, int32_t, EXPECT_EQ, _cth_test_div);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_binary(op, int64_t, EXPECT_EQ, _cth_test_div);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_binary(op, uint8_t, EXPECT_EQ, _cth_test_div);
  }
}

TEST(cTorchDivOpTest, testFloat16Default) {
  test_div(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchDivOpTest, testFloat32Default) {
  test_div(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchDivOpTest, testFloat32MKL) {
  test_div(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchDivOpTest, testFloat64Default) {
  test_div(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchDivOpTest, testFloat64MKL) {
  test_div(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

#ifdef BACKEND_APPLE
TEST(cTorchDivOpTest, testFloat32Apple) {
  test_div(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
}

TEST(cTorchDivOpTest, testFloat64Apple) {
  test_div(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
}
#endif

#ifdef BACKEND_CUDA
TEST(cTorchDivOpTest, testFloat32CUDA) {
  test_div(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchDivOpTest, testFloat64CUDA) {
  test_div(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}
#endif

TEST(cTorchDivOpTest, testInt16Default) {
  test_div(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchDivOpTest, testInt32Default) {
  test_div(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchDivOpTest, testInt64Default) {
  test_div(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}

TEST(cTorchDivOpTest, testUInt8Default) {
  test_div(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 100.0);
}
