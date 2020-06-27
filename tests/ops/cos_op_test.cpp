#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_cos(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
              float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_cos, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;
  op_cos_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, cos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, cos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, cos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, cos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, cos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, cos);
  }
}

TEST(cTorchCosOpTest, testFloat16Default) {
  test_cos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -100.0, 100.0);
}

TEST(cTorchCosOpTest, testFloat32Default) {
  test_cos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(cTorchCosOpTest, testFloat64Default) {
  test_cos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}

TEST(cTorchCosOpTest, testInt16Default) {
  test_cos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -100.0, 100.0);
}

TEST(cTorchCosOpTest, testInt32Default) {
  test_cos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -100.0, 100.0);
}

TEST(cTorchCosOpTest, testInt64Default) {
  test_cos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -100.0, 100.0);
}

TEST(cTorchCosOpTest, testUInt8Default) {
  test_cos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, -100.0, 100.0);
}