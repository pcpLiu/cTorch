#include <tgmath.h>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_fmod(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
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
    op_fmod_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_fmod_mkl(op);
  } else if (backend == CTH_BACKEND_APPLE) {
    op_fmod_apple(op);
  }

  sample_print_triple(
      data_type, array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
      array_at(CTorchTensor)(op->in_bound_tensors, 1)->values,
      array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_binary(op, float, EXPECT_FLOAT_EQ, fmod);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_binary(op, double, EXPECT_DOUBLE_EQ, fmod);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_binary(op, int16_t, EXPECT_EQ, fmod);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_binary(op, int32_t, EXPECT_EQ, fmod);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_binary(op, int64_t, EXPECT_EQ, fmod);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_binary(op, uint8_t, EXPECT_EQ, fmod);
  }
}

TEST(cTorchFmodOpTest, testFloat16Default) {
  test_fmod(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testFloat32Default) {
  test_fmod(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testFloat32MKL) {
  test_fmod(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testFloat64Default) {
  test_fmod(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testFloat64MKL) {
  test_fmod(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testFloat32Apple) {
  test_fmod(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testFloat64Apple) {
  test_fmod(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testInt16Default) {
  test_fmod(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testInt32Default) {
  test_fmod(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testInt64Default) {
  test_fmod(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -100.0, 100.0);
}

TEST(cTorchFmodOpTest, testUInt8Default) {
  test_fmod(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, -100.0, 100.0);
}
