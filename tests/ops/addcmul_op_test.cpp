#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_addcmul(dtype, input, tensor_1, tensor_2, multiplier, output,  \
                        expect_eq)                                             \
  do {                                                                         \
    cth_tensor_dim_t N = input->meta_info->n_elements;                         \
    dtype *input_t = (dtype *)input->values;                                   \
    dtype *tensor_1_t = (dtype *)tensor_1->values;                             \
    dtype *tensor_2_t = (dtype *)tensor_2->values;                             \
    dtype *output_t = (dtype *)output->values;                                 \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      dtype expect_val =                                                       \
          input_t[i] + multiplier * tensor_1_t[i] * tensor_2_t[i];             \
      expect_eq(expect_val, output_t[i]);                                      \
    }                                                                          \
  } while (0);

void test_addcmul(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                  float min, float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = 2;
  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_add, 3, 1, 1);

  CTHTensor *input = create_dummy_tensor(dims, n_dim, data_type, min, max);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);

  CTHTensor *tensor_1 = create_dummy_tensor(dims, n_dim, data_type, min, max);
  tensor_1->meta_info->tensor_name = "tensor_1";
  cth_array_set(CTHTensor)(op->in_bound_tensors, 1, tensor_1);

  CTHTensor *tensor_2 = create_dummy_tensor(dims, n_dim, data_type, min, max);
  tensor_2->meta_info->tensor_name = "tensor_2";
  cth_array_set(CTHTensor)(op->in_bound_tensors, 2, tensor_2);

  CTHTensor *output = create_dummy_tensor(dims, n_dim, data_type, min, max);
  cth_array_set(CTHTensor)(op->out_bound_tensors, 0, output);

  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_MULTIPLIER;
  float multiplier = 0.5;
  param->data.multiplier = &multiplier;
  cth_array_set(CTHParam)(op->params, 0, param);

  op_addcmul_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_addcmul(float, input, tensor_1, tensor_2, 0.5, output,
                    EXPECT_FLOAT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_addcmul(double, input, tensor_1, tensor_2, 0.5, output,
                    EXPECT_DOUBLE_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_addcmul(int16_t, input, tensor_1, tensor_2, 0.5, output, EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_addcmul(int32_t, input, tensor_1, tensor_2, 0.5, output, EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_addcmul(int64_t, input, tensor_1, tensor_2, 0.5, output, EXPECT_EQ)
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_addcmul(uint8_t, input, tensor_1, tensor_2, 0.5, output, EXPECT_EQ)
  }
}

TEST(cTorchAddcmulOpTest, testFloat16Default) {
  test_addcmul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchAddcmulOpTest, testFloat32Default) {
  test_addcmul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchAddcmulOpTest, testFloat64Default) {
  test_addcmul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchAddcmulOpTest, testInt16Default) {
  test_addcmul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchAddcmulOpTest, testInt32Default) {
  test_addcmul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchAddcmulOpTest, testInt64Default) {
  test_addcmul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}

TEST(cTorchAddcmulOpTest, testUInt8Default) {
  test_addcmul(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 100.0);
}
