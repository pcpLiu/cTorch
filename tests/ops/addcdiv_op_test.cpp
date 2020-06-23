#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_addcdiv(dtype, input, tensor_1, tensor_2, value, output,       \
                        expect_eq)                                             \
  do {                                                                         \
    tensor_size_t N = input->meta_info->n_elements;                            \
    dtype *input_t = (dtype *)input->values;                                   \
    dtype *tensor_1_t = (dtype *)tensor_1->values;                             \
    dtype *tensor_2_t = (dtype *)tensor_2->values;                             \
    dtype *output_t = (dtype *)output->values;                                 \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      dtype expect_val = input_t[i] + value * (tensor_1_t[i] / tensor_2_t[i]); \
      expect_eq(expect_val, output_t[i]);                                      \
    }                                                                          \
  } while (0);

void test_addcdiv(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                  float min, float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = 2;
  CTorchOperator *op = create_dummy_op(CTH_OP_ID_add, 4, 1);

  CTorchTensor *input = create_dummy_tensor(dims, n_dim, data_type, min, max);
  array_set(CTorchTensor)(op->in_bound_tensors, 0, input);

  CTorchTensor *tensor_1 =
      create_dummy_tensor(dims, n_dim, data_type, min, max);
  tensor_1->meta_info->tensor_name = "tensor_1";
  array_set(CTorchTensor)(op->in_bound_tensors, 1, tensor_1);

  CTorchTensor *tensor_2 =
      create_dummy_tensor(dims, n_dim, data_type, min, max);
  tensor_2->meta_info->tensor_name = "tensor_2";
  array_set(CTorchTensor)(op->in_bound_tensors, 2, tensor_2);

  tensor_dim_t dims_2[] = {1};
  CTorchTensor *tensor_val =
      create_dummy_tensor(dims_2, 1, CTH_TENSOR_DATA_TYPE_FLOAT_32, min, max);
  tensor_val->meta_info->tensor_name = "value";
  array_set(CTorchTensor)(op->in_bound_tensors, 3, tensor_val);
  float value = ((float *)tensor_val->values)[0];

  CTorchTensor *output = create_dummy_tensor(dims, n_dim, data_type, min, max);
  array_set(CTorchTensor)(op->out_bound_tensors, 0, output);

  op_addcdiv_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_addcdiv(float, input, tensor_1, tensor_2, value, output,
                    EXPECT_FLOAT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_addcdiv(double, input, tensor_1, tensor_2, value, output,
                    EXPECT_DOUBLE_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_addcdiv(int16_t, input, tensor_1, tensor_2, value, output,
                    EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_addcdiv(int32_t, input, tensor_1, tensor_2, value, output,
                    EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_addcdiv(int64_t, input, tensor_1, tensor_2, value, output,
                    EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_addcdiv(uint8_t, input, tensor_1, tensor_2, value, output,
                    EXPECT_EQ)
  }
}

TEST(cTorchAddcdivOpTest, testFloat16Default) {
  test_addcdiv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchAddcdivOpTest, testFloat32Default) {
  test_addcdiv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchAddcdivOpTest, testFloat64Default) {
  test_addcdiv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchAddcdivOpTest, testInt16Default) {
  test_addcdiv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchAddcdivOpTest, testInt32Default) {
  test_addcdiv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchAddcdivOpTest, testInt64Default) {
  test_addcdiv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}

TEST(cTorchAddcdivOpTest, testUInt8Default) {
  test_addcdiv(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 100.0);
}
