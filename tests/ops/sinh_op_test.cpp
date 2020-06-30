#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"
#include <ctgmath>

void test_sinh(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_sinh, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;
  op_sinh_cpu(op);

  sample_print(data_type,
               array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
               array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, sinh);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, sinh);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, sinh);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, sinh);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, sinh);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, sinh);
  }
}

TEST(cTorchSinhOpTest, testFloat16Default) {
  test_sinh(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -20.0, 20.0);
}

TEST(cTorchSinhOpTest, testFloat32Default) {
  test_sinh(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -20.0, 20.0);
}

TEST(cTorchSinhOpTest, testFloat64Default) {
  test_sinh(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -20.0, 20.0);
}

TEST(cTorchSinhOpTest, testInt16Default) {
  test_sinh(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -20.0, 20.0);
}

TEST(cTorchSinhOpTest, testInt32Default) {
  test_sinh(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -20.0, 20.0);
}

TEST(cTorchSinhOpTest, testInt64Default) {
  test_sinh(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -20.0, 20.0);
}

TEST(cTorchSinhOpTest, testUInt8Default) {
  test_sinh(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 20.0);
}
