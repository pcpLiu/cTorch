#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"
#include <tgmath.h>

#define _cth_test_pow(a, b) pow(a, b)

void test_pow(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
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
    op_pow_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
#ifdef BACKEND_MKL
    op_pow_mkl(op);
#endif
  } else if (backend == CTH_BACKEND_APPLE) {
#ifdef BACKEND_APPLE
    op_pow_apple(op);
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    op_pow_cuda(op);
#endif
  }

  sample_print_triple(
      data_type, cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->values,
      cth_array_at(CTHTensor)(op->in_bound_tensors, 1)->values,
      cth_array_at(CTHTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_binary(op, float, EXPECT_FLOAT_EQ, _cth_test_pow);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_binary(op, double, EXPECT_DOUBLE_EQ, _cth_test_pow);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_binary(op, int16_t, EXPECT_EQ, _cth_test_pow);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_binary(op, int32_t, EXPECT_EQ, _cth_test_pow);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_binary(op, int64_t, EXPECT_EQ, _cth_test_pow);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_binary(op, uint8_t, EXPECT_EQ, _cth_test_pow);
  }
}

TEST(cTorchPowOpTest, testFloat16Default) {
  test_pow(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 10);
}

TEST(cTorchPowOpTest, testFloat32Default) {
  test_pow(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10);
}

TEST(cTorchPowOpTest, testFloat64Default) {
  test_pow(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 10);
}

#ifdef BACKEND_MKL
TEST(cTorchPowOpTest, testFloat32MKL) {
  test_pow(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10);
}

TEST(cTorchPowOpTest, testFloat64MKL) {
  test_pow(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 10);
}
#endif

#ifdef BACKEND_APPLE
TEST(cTorchPowOpTest, testFloat32Apple) {
  test_pow(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10);
}

TEST(cTorchPowOpTest, testFloat64Apple) {
  test_pow(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 10);
}
#endif

#ifdef BACKEND_CUDA
TEST(cTorchPowOpTest, testFloat32CUDA) {
  test_pow(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchPowOpTest, testFloat64CUDA) {
  test_pow(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}
#endif

TEST(cTorchPowOpTest, testInt16Default) {
  test_pow(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 10);
}

TEST(cTorchPowOpTest, testInt32Default) {
  test_pow(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 10);
}

TEST(cTorchPowOpTest, testInt64Default) {
  test_pow(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 10);
}

TEST(cTorchPowOpTest, testUInt8Default) {
  test_pow(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 10);
}
