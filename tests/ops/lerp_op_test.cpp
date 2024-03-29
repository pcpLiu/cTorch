#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_lerp(op, data_type, expect_fn)                                 \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *input_3 = cth_array_at(CTHTensor)(op->in_bound_tensors, 2);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *input_1_t = (data_type *)input_1->values;                       \
    data_type *input_2_t = (data_type *)input_2->values;                       \
    data_type *input_3_t = (data_type *)input_3->values;                       \
    data_type *output_t = (data_type *)output->values;                         \
    for (cth_tensor_dim_t i = 0; i < input_1->meta_info->n_elements; i++) {    \
      data_type val =                                                          \
          input_1_t[i] + input_3_t[i] * (input_2_t[i] - input_1_t[i]);         \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
                                                                               \
  } while (0)

void test_lerp(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHOperator *op = create_dummy_op(CTH_OP_ID_add, 3, 1);
  cth_array_set(CTHTensor)(
      op->in_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->in_bound_tensors, 1,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->in_bound_tensors, 2,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  cth_array_set(CTHTensor)(
      op->out_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  op_lerp_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_lerp(op, float, EXPECT_FLOAT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_lerp(op, double, EXPECT_DOUBLE_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_lerp(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_lerp(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_lerp(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_lerp(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_lerp(op, bool, EXPECT_EQ);
  }
}

TEST(cTorchLerpOpTest, testFloat16Default) {
  test_lerp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchLerpOpTest, testFloat32Default) {
  test_lerp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchLerpOpTest, testFloat64Default) {
  test_lerp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchLerpOpTest, testInt16Default) {
  test_lerp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchLerpOpTest, testInt32Default) {
  test_lerp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchLerpOpTest, testInt64Default) {
  test_lerp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}

TEST(cTorchLerpOpTest, testUInt8Default) {
  test_lerp(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 100.0);
}
