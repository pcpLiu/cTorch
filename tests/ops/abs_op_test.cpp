#include <tgmath.h>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_abs(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
              float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_abs, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_abs_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_abs_mkl(op);
  } else if (backend == CTH_BACKEND_APPLE) {
    op_abs_apple(op);
  }

  sample_print(data_type,
               array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
               array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, fabs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, fabs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, fabs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, fabs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, fabs);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, fabs);
  }
}

TEST(cTorchAbsOpTest, testFloat16Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testFloat32Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testFloat32MKL) {
  test_abs(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testFloat64Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testFloat64MKL) {
  test_abs(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testFloat32Apple) {
  test_abs(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testFloat64Apple) {
  test_abs(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testInt16Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testInt32Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testInt64Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -1.0, 1.0);
}

TEST(cTorchAbsOpTest, testUInt8Default) {
  test_abs(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, -1.0, 1.0);
}
