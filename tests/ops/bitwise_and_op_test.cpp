#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_bitwise_and(op, data_type, expect_fn)                          \
  do {                                                                         \
    CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *input_1_t = (data_type *)input_1->values;                       \
    data_type *input_2_t = (data_type *)input_2->values;                       \
    data_type *output_t = (data_type *)output->values;                         \
                                                                               \
    for (int i = 0; i < input_1->meta_info->n_elements; i++) {                 \
      data_type val = input_1_t[i] & input_2_t[i];                             \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
  } while (0)

void test_bitwise_and(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                      float min, float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = 2;
  CTorchOperator *op = create_dummy_op(CTH_OP_ID_add, 2, 1);
  array_set(CTorchTensor)(
      op->in_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(
      op->in_bound_tensors, 1,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(
      op->out_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));

  op_bitwise_and_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_bitwise_and(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_bitwise_and(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_bitwise_and(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_bitwise_and(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_bitwise_and(op, bool, EXPECT_EQ);
  }
}

TEST(cTorchBitwiseAndOpTest, testInt16Default) {
  test_bitwise_and(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0,
                   100.0);
}

TEST(cTorchBitwiseAndOpTest, testInt32Default) {
  test_bitwise_and(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0,
                   100.0);
}

TEST(cTorchBitwiseAndOpTest, testInt64Default) {
  test_bitwise_and(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0,
                   100.0);
}

TEST(cTorchBitwiseAndOpTest, testUInt8Default) {
  test_bitwise_and(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0,
                   100.0);
}

TEST(cTorchBitwiseAndOpTest, testBoolDefault) {
  test_bitwise_and(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_BOOL, 1.0, 100.0);
}
