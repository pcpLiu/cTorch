#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_acos(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node =
      create_dummy_op_node(CTH_OP_ID_acos, dims, n_dim, data_type, min, max);
  cth_execute_node(op_node, backend);
  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal(float, EXPECT_FLOAT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal(double, EXPECT_DOUBLE_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal(int16_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal(int32_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal(int64_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal(uint8_t, EXPECT_EQ, acos);
  }
}

TEST(acosOpTest, testFloat16Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -1.0, 1.0);
}

TEST(acosOpTest, testFloat32Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(acosOpTest, testFloat64Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(acosOpTest, testInt16Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -1.0, 1.0);
}

TEST(acosOpTest, testInt32Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -1.0, 1.0);
}

TEST(acosOpTest, testInt64Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -1.0, 1.0);
}

TEST(acosOpTest, testUInt8Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 1.0);
}

TEST(acosOpTest, testBoolDefaultExpectExit) {
  CTH_NAN_EXIT = true;
  tensor_dim_t dims[] = {1, 1};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node(CTH_OP_ID_acos, dims, n_dim,
                                             CTH_TENSOR_DATA_TYPE_BOOL, 0, 0);
  EXPECT_EXIT(cth_execute_node(op_node, CTH_BACKEND_DEFAULT),
              ::testing::ExitedWithCode(1),
              "Operator does not support data type");
}
