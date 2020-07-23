#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_true_divide(op, data_type, expect_fn)                          \
  do {                                                                         \
    CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *input_1_t = (data_type *)input_1->values;                       \
    data_type *input_2_t = (data_type *)input_2->values;                       \
    float *output_t = (float *)output->values;                                 \
                                                                               \
    for (tensor_size_t i = 0; i < input_1->meta_info->n_elements; i++) {       \
      float val = (float)input_1_t[i] / (float)input_2_t[i];                   \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
  } while (0)

void test_true_divide(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                      float min, float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = 2;
  CTorchOperator *op = create_dummy_op(CTH_OP_ID_true_divide, 2, 1);
  array_set(CTorchTensor)(
      op->in_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(
      op->in_bound_tensors, 1,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(op->out_bound_tensors, 0,
                          create_dummy_tensor(dims, n_dim,
                                              CTH_TENSOR_DATA_TYPE_FLOAT_32,
                                              min, max));

  op_true_divide_cpu(op);

  sample_print_triple(CTH_TENSOR_DATA_TYPE_FLOAT_32,
                      array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
                      array_at(CTorchTensor)(op->in_bound_tensors, 1)->values,
                      array_at(CTorchTensor)(op->out_bound_tensors, 0)->values,
                      2);

  if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_true_divide(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_true_divide(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_true_divide(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_true_divide(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_true_divide(op, bool, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
             data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_true_divide(op, float, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_true_divide(op, double, EXPECT_EQ);
  }
}

TEST(cTorchTruedivideOpTest, testFloat16Default) {
  test_true_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1, 10.0);
}

TEST(cTorchTruedivideOpTest, testFloat32Default) {
  test_true_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1, 10.0);
}

TEST(cTorchTruedivideOpTest, testFloat64Default) {
  test_true_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1, 10.0);
}

TEST(cTorchTruedivideOpTest, testInt16Default) {
  test_true_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1, 10.0);
}

TEST(cTorchTruedivideOpTest, testInt32Default) {
  test_true_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1, 10.0);
}

TEST(cTorchTruedivideOpTest, testInt64Default) {
  test_true_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1, 10.0);
}

TEST(cTorchTruedivideOpTest, testUInt8Default) {
  test_true_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 10.0);
}
