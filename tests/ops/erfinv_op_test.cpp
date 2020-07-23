#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

/**
 * @note We make the precision (1e-3). Otherwise, it often fails due to
 * precision issues
 */

torch::Tensor _erfinv_torch(torch::Tensor py_tensor) {
  return py_tensor.erfinv();
}

void test_erfinv(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                 float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_erfinv, dims,
                                                   n_dim, data_type, min, max);
  CTorchOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_erfinv_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
#ifdef BACKEND_MKL
    op_erfinv_mkl(op);
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    op_erfinv_cuda(op);
#endif
  }

  sample_print(data_type,
               array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
               array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary_pytorch(op, float, EXPECT_EQ_PRECISION, 1e-3,
                                  _erfinv_torch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary_pytorch(op, double, EXPECT_EQ_PRECISION, 1e-3,
                                  _erfinv_torch);
  }
}

TEST(cTorchErfinvOpTest, testFloat16Default) {
  test_erfinv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -1.0, 1.0);
}

TEST(cTorchErfinvOpTest, testFloat32Default) {
  test_erfinv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchErfinvOpTest, testFloat64Default) {
  test_erfinv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

#ifdef BACKEND_CUDA
TEST(cTorchErfinvOpTest, testFloat32CUDA) {
  test_erfinv(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchErfinvOpTest, testFloat64CUDA) {
  test_erfinv(CTH_BACKEND_CUDA, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}
#endif

#ifdef BACKEND_MKL
TEST(cTorchErfinvOpTest, testFloat32MKL) {
  test_erfinv(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(cTorchErfinvOpTest, testFloat64MKL) {
  test_erfinv(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}
#endif
