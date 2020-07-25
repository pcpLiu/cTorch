#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

/**
 * @note We make the precision (1e-3). Otherwise, it often fails due to
 * precision issues
 */

torch::Tensor _digamma_torch(torch::Tensor py_tensor) {
  return py_tensor.digamma();
}

void test_digamma(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                  float min, float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_digamma, dims, n_dim,
                                                data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_digamma_cpu(op);
  }

  sample_print(data_type,
               cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->values,
               cth_array_at(CTHTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary_pytorch(op, float, EXPECT_EQ_PRECISION, 1e-3,
                                  _digamma_torch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary_pytorch(op, double, EXPECT_EQ_PRECISION, 1e-3,
                                  _digamma_torch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary_pytorch(op, int16_t, EXPECT_EQ_PRECISION, 1e-3,
                                  _digamma_torch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary_pytorch(op, int32_t, EXPECT_EQ_PRECISION, 1e-3,
                                  _digamma_torch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary_pytorch(op, int64_t, EXPECT_EQ_PRECISION, 1e-3,
                                  _digamma_torch);
  }
}

TEST(cTorchDigammaOpTest, testFloat16Default) {
  test_digamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchDigammaOpTest, testFloat32Default) {
  test_digamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchDigammaOpTest, testFloat64Default) {
  test_digamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchDigammaOpTest, testInt16Default) {
  test_digamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchDigammaOpTest, testInt32Default) {
  test_digamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchDigammaOpTest, testInt64Default) {
  test_digamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}
