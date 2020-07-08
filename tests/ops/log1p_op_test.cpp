#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_log1p(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_log1p, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_log1p_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
    op_log1p_mkl(op);
  } else if (backend == CTH_BACKEND_APPLE) {
#ifdef BACKEND_APPLE
    op_log1p_apple(op);
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    op_log1p_cuda(op);
#endif
  }

  sample_print(data_type,
               array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
               array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, log1p);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, log1p);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, log1p);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, log1p);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, log1p);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, log1p);
  }
}

TEST(cTorchLog1pOpTest, testFloat16Default) {
  test_log1p(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testFloat32Default) {
  test_log1p(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testFloat32MKL) {
  test_log1p(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testFloat64Default) {
  test_log1p(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testFloat64MKL) {
  test_log1p(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}

#ifdef BACKEND_APPLE
TEST(cTorchLog1pOpTest, testFloat32Apple) {
  test_log1p(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testFloat64Apple) {
  test_log1p(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}
#endif

#ifdef BACKEND_CUDA
TEST(cTorchLog1pOpTest, testFloat32CUDA) {
  test_log1p(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testFloat64CUDA) {
  test_log1p(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}
#endif

TEST(cTorchLog1pOpTest, testInt16Default) {
  test_log1p(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testInt32Default) {
  test_log1p(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testInt64Default) {
  test_log1p(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 0.01, 20.0);
}

TEST(cTorchLog1pOpTest, testUInt8Default) {
  test_log1p(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 20.0);
}
