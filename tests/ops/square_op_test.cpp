#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"
#include <ctgmath>

#define _cth_square_test_kernel(x) (x * x)

void test_square(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                 float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_square, dims, n_dim,
                                                data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_square_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
#ifdef BACKEND_MKL
    op_square_mkl(op);
#endif
  }

  sample_print(data_type,
               cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->values,
               cth_array_at(CTHTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, _cth_square_test_kernel);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ,
                          _cth_square_test_kernel);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, _cth_square_test_kernel);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, _cth_square_test_kernel);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, _cth_square_test_kernel);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, _cth_square_test_kernel);
  }
}

TEST(cTorchSquareOpTest, testFloat16Default) {
  test_square(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 0.1, 20.0);
}

TEST(cTorchSquareOpTest, testFloat32Default) {
  test_square(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.1, 20.0);
}

TEST(cTorchSquareOpTest, testFloat64Default) {
  test_square(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.1, 20.0);
}

#ifdef BACKEND_MKL
TEST(cTorchSquareOpTest, testFloat32MKL) {
  test_square(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, 0.1, 20.0);
}

TEST(cTorchSquareOpTest, testFloat64MKL) {
  test_square(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, 0.1, 20.0);
}
#endif

TEST(cTorchSquareOpTest, testInt16Default) {
  test_square(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 0.1, 20.0);
}

TEST(cTorchSquareOpTest, testInt32Default) {
  test_square(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 0.1, 20.0);
}

TEST(cTorchSquareOpTest, testInt64Default) {
  test_square(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 0.1, 20.0);
}

TEST(cTorchSquareOpTest, testUInt8Default) {
  test_square(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.1, 10.0);
}
