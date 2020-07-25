#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _cth_test_mul(a, b) (a * b)

void test_mul(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
              float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = 2;
  CTHOperator *op = create_dummy_op(CTH_OP_ID_add, 2, 1);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->in_bound_tensors, 1,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->out_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  if (backend == CTH_BACKEND_DEFAULT) {
    op_mul_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_mul_mkl(op);
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    op_mul_cuda(op);
#endif
  }

  sample_print_triple(
      data_type, cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->values,
      cth_array_at(CTHTensor)(op->in_bound_tensors, 1)->values,
      cth_array_at(CTHTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_binary(op, float, EXPECT_FLOAT_EQ, _cth_test_mul);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_binary(op, double, EXPECT_DOUBLE_EQ, _cth_test_mul);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_binary(op, int16_t, EXPECT_EQ, _cth_test_mul);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_binary(op, int32_t, EXPECT_EQ, _cth_test_mul);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_binary(op, int64_t, EXPECT_EQ, _cth_test_mul);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_binary(op, uint8_t, EXPECT_EQ, _cth_test_mul);
  }
}

TEST(cTorchMulOpTest, testFloat16Default) {
  test_mul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -100.0, 100.0);
}

TEST(cTorchMulOpTest, testFloat32Default) {
  test_mul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(cTorchMulOpTest, testFloat32MKL) {
  test_mul(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(cTorchMulOpTest, testFloat64Default) {
  test_mul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}

TEST(cTorchMulOpTest, testFloat64MKL) {
  test_mul(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}

#ifdef BACKEND_CUDA
TEST(cTorchMulOpTest, testFloat32CUDA) {
  test_mul(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchMulOpTest, testFloat64CUDA) {
  test_mul(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}
#endif

TEST(cTorchMulOpTest, testInt16Default) {
  test_mul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -100.0, 100.0);
}

TEST(cTorchMulOpTest, testInt32Default) {
  test_mul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -100.0, 100.0);
}

TEST(cTorchMulOpTest, testInt64Default) {
  test_mul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -100.0, 100.0);
}

TEST(cTorchMulOpTest, testUInt8Default) {
  test_mul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1, 10);
}
