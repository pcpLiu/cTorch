#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_Hardsigmoid_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {
  // TODO: libtorch does not have hardsigmoid now. Add test later
}

void test_hardsigmoid(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = 2;
  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_Hardsigmoid, 1, 1, 1);

  cth_array_set(CTHTensor)(
      op->in_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->out_bound_tensors,
      0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  if (backend == CTH_BACKEND_DEFAULT) {
    op_Hardsigmoid_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _Hardsigmoid_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _Hardsigmoid_pytorch);
  }
}

// TEST(cTorchHardsigmoidOpTest, testFloat32Default) {
//   test_hardsigmoid(
//       CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
// }

// TEST(cTorchHardsigmoidOpTest, testFloat64Default) {
//   test_hardsigmoid(
//       CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
// }
