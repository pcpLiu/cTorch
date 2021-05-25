#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
__mean_pytorch(torch::Tensor &pytorch_in_tensor, cth_tensor_dim_t reduce_dim) {
  if (reduce_dim == -1) {
    return pytorch_in_tensor.mean();
  } else {
    return pytorch_in_tensor.mean(reduce_dim, false);
  }
}

void test_mean(
    CTH_BACKEND backend,
    CTH_TENSOR_DATA_TYPE data_type,
    float min,
    float max,
    bool flat) {
  cth_tensor_dim_t n_dim = 5, min_dim = 1, max_dim = 10;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * n_dim);
  _rand_dims(dims, n_dim, min_dim, max_dim);

  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_mean, 1, 1, 1);

  CTHTensor *input = create_dummy_tensor(dims, n_dim, data_type, min, max);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);

  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  cth_tensor_dim_t dim = _rand_int(0, n_dim - 1);
  param->data.dim_val = &dim;
  if (flat) {
    *(param->data.dim_val) = -1;
  }
  param->type = CTH_PARAM_TYPE_DIM;
  cth_array_set(CTHParam)(op->params, 0, param);

  cth_tensor_dim_t n_dim_out;
  if (flat) {
    n_dim_out = 1;
  } else {
    n_dim_out = n_dim - 1;
  }
  cth_tensor_dim_t *out_dims =
      (cth_tensor_dim_t *)MALLOC(sizeof(cth_tensor_dim_t) * n_dim_out);
  if (flat) {
    out_dims[0] = 1;
  } else {
    _get_reduce_dims(dims, n_dim, *(param->data.dim_val), out_dims);
  }
  CTHTensor *output =
      create_dummy_tensor(out_dims, n_dim_out, data_type, min, max);
  cth_array_set(CTHTensor)(op->out_bound_tensors, 0, output);

  if (backend == CTH_BACKEND_DEFAULT) {
    op_mean_cpu(op);
  }

  _reduce_typing_test_flow(
      op,
      data_type,
      data_type,
      float,
      __mean_pytorch,
      EXPECT_EQ_PRECISION_0001);
}

TEST(cTorchMeanOpTest, testFloat16Default) {
  test_mean(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -10.0, 10.0, false);
}

TEST(cTorchMeanOpTest, testFloat32Default) {
  test_mean(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0, false);
}

TEST(cTorchMeanOpTest, testFloat64Default) {
  test_mean(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0, false);
}

TEST(cTorchMeanOpTest, testFloat64DefaultFlat) {
  test_mean(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0, true);
}

TEST(cTorchMeanOpTest, testInt16Default) {
  test_mean(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10, 10, false);
}

TEST(cTorchMeanOpTest, testInt32Default) {
  test_mean(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -10, 10, false);
}

TEST(cTorchMeanOpTest, testInt64Default) {
  test_mean(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -10, 10, false);
}

TEST(cTorchMeanOpTest, testUInt8Default) {
  test_mean(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1, 10, false);
}
