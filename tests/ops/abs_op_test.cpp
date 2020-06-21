#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_abs(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
              float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node =
      create_dummy_op_node(CTH_OP_ID_abs, dims, n_dim, data_type, min, max);
  cth_execute_node(op_node, backend);
  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(float, EXPECT_FLOAT_EQ, fabs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(double, EXPECT_DOUBLE_EQ, fabs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(int16_t, EXPECT_EQ, abs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(int32_t, EXPECT_EQ, abs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(int64_t, EXPECT_EQ, abs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(uint8_t, EXPECT_EQ, abs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _ele_wise_equal_unary(bool, EXPECT_EQ, abs);
  }
}

TEST(absOpTest, testFloat16Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -100.0, 100.0);
}

TEST(absOpTest, testFloat32Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(absOpTest, testFloat64Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}

TEST(absOpTest, testInt16Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -100.0, 100.0);
}

TEST(absOpTest, testInt32Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -100.0, 100.0);
}

TEST(absOpTest, testInt64Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -100.0, 100.0);
}

TEST(absOpTest, testUInt8Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, -100.0, 100.0);
}

TEST(absOpTest, testBoolDefault) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_BOOL, -100.0, 100.0);
}