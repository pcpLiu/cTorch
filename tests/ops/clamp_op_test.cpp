#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_clamp(dtype, input, min, max, output, expect_eq)               \
  do {                                                                         \
    tensor_size_t N = input->meta_info->n_elements;                            \
    dtype *input_t = (dtype *)input->values;                                   \
    dtype *output_t = (dtype *)output->values;                                 \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      dtype expect_val;                                                        \
      if (input_t[i] < min) {                                                  \
        expect_val = min;                                                      \
      } else if (input_t[i] > max) {                                           \
        expect_val = max;                                                      \
      } else {                                                                 \
        expect_val = input_t[i];                                               \
      }                                                                        \
      expect_eq(expect_val, output_t[i]);                                      \
    }                                                                          \
  } while (0);

void test_clamp(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
                float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = 2;
  CTorchOperator *op = create_dummy_op_with_param(CTH_OP_ID_add, 1, 1, 2);

  CTorchTensor *input = create_dummy_tensor(dims, n_dim, data_type, min, max);
  array_set(CTorchTensor)(op->in_bound_tensors, 0, input);

  CTorchTensor *output = create_dummy_tensor(dims, n_dim, data_type, min, max);
  array_set(CTorchTensor)(op->out_bound_tensors, 0, output);

  CTorchParam *param = (CTorchParam *)MALLOC(sizeof(CTorchParam));
  param->type = CTH_PARAM_TYPE_MAX_FLOAT32;
  max = _rand_float(-10, 10);
  param->data.max = max;
  array_set(CTorchParam)(op->params, 0, param);

  CTorchParam *param2 = (CTorchParam *)MALLOC(sizeof(CTorchParam));
  param2->type = CTH_PARAM_TYPE_MIN_FLOAT32;
  min = _rand_float(-10, 10);
  param2->data.min = min;
  array_set(CTorchParam)(op->params, 1, param2);

  op_clamp_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_clamp(float, input, min, max, output, EXPECT_FLOAT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_clamp(double, input, min, max, output, EXPECT_DOUBLE_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_clamp(int16_t, input, min, max, output, EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_clamp(int32_t, input, min, max, output, EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_clamp(int64_t, input, min, max, output, EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_clamp(uint8_t, input, min, max, output, EXPECT_EQ)
  }
}

TEST(cTorchClampOpTest, testFloat16Default) {
  test_clamp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchClampOpTest, testFloat32Default) {
  test_clamp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchClampOpTest, testFloat64Default) {
  test_clamp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchClampOpTest, testInt16Default) {
  test_clamp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchClampOpTest, testInt32Default) {
  test_clamp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchClampOpTest, testInt64Default) {
  test_clamp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}

TEST(cTorchClampOpTest, testUInt8Default) {
  test_clamp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 100.0);
}
