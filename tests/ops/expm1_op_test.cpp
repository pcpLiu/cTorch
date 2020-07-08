#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_expm1(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_expm1, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_expm1_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_expm1_mkl(op);
  } else if (backend == CTH_BACKEND_APPLE) {
#ifdef BACKEND_APPLE
    op_expm1_apple(op);
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    op_expm1_cuda(op);
#endif
  }

  sample_print(data_type,
               array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
               array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, expm1);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, expm1);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, expm1);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, expm1);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, expm1);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, expm1);
  }
}

TEST(cTorchExpm1OpTest, testFloat16Default) {
  test_expm1(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testFloat32Default) {
  test_expm1(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testFloat32MKL) {
  test_expm1(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testFloat64Default) {
  test_expm1(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testFloat64MKL) {
  test_expm1(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -20.0, 20.0);
}

#ifdef BACKEND_APPLE
TEST(cTorchExpm1OpTest, testFloat32Apple) {
  test_expm1(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testFloat64Apple) {
  test_expm1(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, -20.0, 20.0);
}
#endif

#ifdef BACKEND_CUDA
TEST(cTorchExpm1OpTest, testFloat32CUDA) {
  test_expm1(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchExpm1OpTest, testFloat64CUDA) {
  test_expm1(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}
#endif

TEST(cTorchExpm1OpTest, testInt16Default) {
  test_expm1(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testInt32Default) {
  test_expm1(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testInt64Default) {
  test_expm1(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -20.0, 20.0);
}

TEST(cTorchExpm1OpTest, testUInt8Default) {
  test_expm1(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 20.0);
}
