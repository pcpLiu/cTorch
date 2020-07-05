#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _test_atan2(a, b) atan2(b, a)

void test_atan2(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
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
    op_atan2_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_atan2_mkl(op);
  } else if (backend == CTH_BACKEND_APPLE) {
    op_atan2_apple(op);
  }

  sample_print_triple(
      data_type, array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
      array_at(CTorchTensor)(op->in_bound_tensors, 1)->values,
      array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_binary(op, float, EXPECT_FLOAT_EQ, _test_atan2);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_binary(op, double, EXPECT_DOUBLE_EQ, _test_atan2);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_binary(op, int16_t, EXPECT_EQ, _test_atan2);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_binary(op, int32_t, EXPECT_EQ, _test_atan2);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_binary(op, int64_t, EXPECT_EQ, _test_atan2);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_binary(op, uint8_t, EXPECT_EQ, _test_atan2);
  }
}

TEST(cTorchAtan2OpTest, testFloat16Default) {
  test_atan2(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testFloat32Default) {
  test_atan2(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testFloat32MKL) {
  test_atan2(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testFloat64Default) {
  test_atan2(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testFloat64MKL) {
  test_atan2(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testFloat32Apple) {
  test_atan2(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testFloat64Apple) {
  test_atan2(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testInt16Default) {
  test_atan2(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testInt32Default) {
  test_atan2(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testInt64Default) {
  test_atan2(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -1.0, 1.0);
}

TEST(cTorchAtan2OpTest, testUInt8Default) {
  test_atan2(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 1.0);
}
