#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_round(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_round, dims, n_dim,
                                                data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_round_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
#ifdef BACKEND_MKL
    op_round_mkl(op);
#endif
  } else if (backend == CTH_BACKEND_APPLE) {
#ifdef BACKEND_APPLE
    op_round_apple(op);
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    op_round_cuda(op);
#endif
  }

  sample_print(data_type,
               cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->values,
               cth_array_at(CTHTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, round);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, round);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, round);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, round);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, round);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, round);
  }
}

TEST(cTorchRoundOpTest, testFloat16Default) {
  test_round(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 0.01, 20.0);
}

TEST(cTorchRoundOpTest, testFloat32Default) {
  test_round(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchRoundOpTest, testFloat64Default) {
  test_round(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}

#ifdef BACKEND_MKL
TEST(cTorchRoundOpTest, testFloat32MKL) {
  test_round(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchRoundOpTest, testFloat64MKL) {
  test_round(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}
#endif

#ifdef BACKEND_APPLE
TEST(cTorchRoundOpTest, testFloat32Apple) {
  test_round(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.01, 20.0);
}

TEST(cTorchRoundOpTest, testFloat64Apple) {
  test_round(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.01, 20.0);
}
#endif

#ifdef BACKEND_CUDA
TEST(cTorchRoundOpTest, testFloat32CUDA) {
  test_round(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0, 100.0);
}

TEST(cTorchRoundOpTest, testFloat64CUDA) {
  test_round(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0, 100.0);
}
#endif

TEST(cTorchRoundOpTest, testInt16Default) {
  test_round(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 0.01, 20.0);
}

TEST(cTorchRoundOpTest, testInt32Default) {
  test_round(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 0.01, 20.0);
}

TEST(cTorchRoundOpTest, testInt64Default) {
  test_round(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 0.01, 20.0);
}

TEST(cTorchRoundOpTest, testUInt8Default) {
  test_round(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 20.0);
}
