#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

/**
 * This test sometimes failed on int cases
 *
 */

#define _verify_dist(data_type, output, p, expect_eq, dtype_enum)              \
  do {                                                                         \
    data_type *output_t = (data_type *)output->values;                         \
                                                                               \
    torch::Tensor input_1_py =                                                 \
        create_torch_tensor(cth_array_at(CTHTensor)(op->in_bound_tensors, 0)); \
    torch::Tensor input_2_py =                                                 \
        create_torch_tensor(cth_array_at(CTHTensor)(op->in_bound_tensors, 1)); \
    auto out_py = input_1_py.dist(input_2_py, p);                              \
    auto expect_result = (float *)out_py.data_ptr();                           \
    data_type expect_val = expect_result[0];                                   \
    if (dtype_enum == CTH_TENSOR_DATA_TYPE_INT_16 ||                           \
        dtype_enum == CTH_TENSOR_DATA_TYPE_INT_32 ||                           \
        dtype_enum == CTH_TENSOR_DATA_TYPE_INT_64 ||                           \
        dtype_enum == CTH_TENSOR_DATA_TYPE_UINT_8) {                           \
      expect_val = round(expect_result[0]);                                    \
    }                                                                          \
                                                                               \
    expect_eq(expect_val, output_t[0], 1e-2);                                  \
  } while (0);

void test_dist(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  cth_tensor_dim_t n_dim = 2;
  cth_tensor_dim_t dims[] = {100, 100};

  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_dist, 2, 1, 1);

  CTHTensor *input = create_dummy_tensor(dims, n_dim, data_type, min, max);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);

  CTHTensor *input2 = create_dummy_tensor(dims, n_dim, data_type, min, max);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 1, input2);

  cth_tensor_dim_t out_dims[] = {1};
  CTHTensor *output = create_dummy_tensor(out_dims, 1, data_type, min, max);
  cth_array_set(CTHTensor)(op->out_bound_tensors, 0, output);

  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  float p = _rand_float(1, 10);
  param->data.p = p;
  param->type = CTH_PARAM_TYPE_P_FLOAT32;
  cth_array_set(CTHParam)(op->params, 0, param);

  if (backend == CTH_BACKEND_DEFAULT) {
    op_dist_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_dist(float, output, p, EXPECT_EQ_PRECISION, data_type)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_dist(double, output, p, EXPECT_EQ_PRECISION, data_type)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_dist(int16_t, output, p, EXPECT_EQ_PRECISION, data_type)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_dist(int32_t, output, p, EXPECT_EQ_PRECISION, data_type)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_dist(int64_t, output, p, EXPECT_EQ_PRECISION, data_type)
  }
}

TEST(cTorchDistOpTest, testFloat16Default) {
  test_dist(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 10.0);
}

TEST(cTorchDistOpTest, testFloat32Default) {
  test_dist(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
}

TEST(cTorchDistOpTest, testFloat64Default) {
  test_dist(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 10.0);
}

// TODO: int test often failed randomlly

// TEST(cTorchDistOpTest, testInt32Default) {
//   test_dist(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
// }

// TEST(cTorchDistOpTest, testInt64Default) {
//   test_dist(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
// }
