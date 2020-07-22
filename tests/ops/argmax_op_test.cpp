#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor __argmax_pytorch(torch::Tensor &pytorch_in_tensor,
                               tensor_dim_t reduce_dim) {
  return pytorch_in_tensor.argmax(reduce_dim, false);
}

void test_argmax(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                 float max) {
  tensor_dim_t n_dim = 5, min_dim = 1, max_dim = 10;
  tensor_dim_t *dims = (tensor_dim_t *)MALLOC(sizeof(tensor_dim_t) * n_dim);
  _rand_dims(dims, n_dim, min_dim, max_dim);

  CTorchOperator *op = create_dummy_op_with_param(CTH_OP_ID_argmax, 1, 1, 1);

  CTorchTensor *input = create_dummy_tensor(dims, n_dim, data_type, min, max);
  array_set(CTorchTensor)(op->in_bound_tensors, 0, input);

  CTorchParam *param = (CTorchParam *)MALLOC(sizeof(CTorchParam));
  param->data.dim = _rand_int(0, n_dim - 1);
  param->type = CTH_PARAM_TYPE_DIM_INT32;
  array_set(CTorchParam)(op->params, 0, param);

  tensor_dim_t n_dim_out = n_dim - 1;
  tensor_dim_t *out_dims =
      (tensor_dim_t *)MALLOC(sizeof(tensor_dim_t) * n_dim_out);
  _get_reduce_dims(dims, n_dim, param->data.dim, out_dims);
  CTorchTensor *output = create_dummy_tensor(
      out_dims, n_dim_out, CTH_TENSOR_DATA_TYPE_INT_64, min, max);
  array_set(CTorchTensor)(op->out_bound_tensors, 0, output);

  if (backend == CTH_BACKEND_DEFAULT) {
    op_argmax_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _reduce_op(op, float, int64_t, __argmax_pytorch, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _reduce_op(op, double, int64_t, __argmax_pytorch, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _reduce_op(op, int16_t, int64_t, __argmax_pytorch, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _reduce_op(op, int32_t, int64_t, __argmax_pytorch, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _reduce_op(op, int64_t, int64_t, __argmax_pytorch, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _reduce_op(op, uint8_t, int64_t, __argmax_pytorch, EXPECT_EQ);
  }
}

TEST(cTorchArgmaxOpTest, testFloat16Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -100.0,
              100.0);
}

TEST(cTorchArgmaxOpTest, testFloat32Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -100.0,
              100.0);
}

TEST(cTorchArgmaxOpTest, testFloat64Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -100.0,
              100.0);
}

TEST(cTorchArgmaxOpTest, testInt16Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0, 10.0);
}

TEST(cTorchArgmaxOpTest, testInt32Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -100.0, 100.0);
}

TEST(cTorchArgmaxOpTest, testInt64Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -100.0, 100.0);
}

TEST(cTorchArgmaxOpTest, testUInt8Default) {
  test_argmax(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0, 10.0);
}
