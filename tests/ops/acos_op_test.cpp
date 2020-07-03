#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_acos(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_acos, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_acos_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_acos_mkl(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, acos);
  }
}

TEST(cTorchAcosOpTest, testFloat16Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testFloat32Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testFloat32MKL) {
  test_acos(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testFloat64Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testFloat64MKL) {
  test_acos(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testInt16Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testInt32Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testInt64Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -1.0, 1.0);
}

TEST(cTorchAcosOpTest, testUInt8Default) {
  test_acos(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 1.0);
}

TEST(cTorchAcosOpTest, testBoolDefaultExpectExit) {
  CTH_NAN_EXIT = true;
  tensor_dim_t dims[] = {1, 1};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(
      CTH_OP_ID_acos, dims, n_dim, CTH_TENSOR_DATA_TYPE_BOOL, 0, 0);
  EXPECT_EXIT(cth_execute_node(op_node, CTH_BACKEND_DEFAULT),
              ::testing::ExitedWithCode(1),
              "Operator does not support data type");
}
