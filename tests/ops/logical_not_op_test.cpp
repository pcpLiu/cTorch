#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_logical_and(op, data_type, expect_fn)                          \
  do {                                                                         \
    CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);     \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *input_t = (data_type *)input->values;                           \
    bool *output_t = (bool *)output->values;                                   \
                                                                               \
    for (int i = 0; i < input->meta_info->n_elements; i++) {                   \
      uint16_t x = (0 == input_t[i] ? 0 : 1);                                  \
      bool val = (bool)(1 - x);                                                \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
  } while (0)

void test_logical_not(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                      float min, float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = 2;
  CTorchOperator *op = create_dummy_op(CTH_OP_ID_add, 1, 1);
  array_set(CTorchTensor)(
      op->in_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, data_type, min, max));
  array_set(CTorchTensor)(
      op->out_bound_tensors, 0,
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_BOOL, min, max));

  op_logical_not_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_logical_and(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_logical_and(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_logical_and(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_logical_and(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_logical_and(op, bool, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
             data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_logical_and(op, float, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_logical_and(op, double, EXPECT_EQ);
  }
}

TEST(cTorchLogicalNotOpTest, testFloat16Default) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -10.0,
                   10.0);
}

TEST(cTorchLogicalNotOpTest, testFloat32Default) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0,
                   10.0);
}

TEST(cTorchLogicalNotOpTest, testFloat64Default) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0,
                   10.0);
}

TEST(cTorchLogicalNotOpTest, testInt16Default) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0,
                   10.0);
}

TEST(cTorchLogicalNotOpTest, testInt32Default) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -10.0,
                   10.0);
}

TEST(cTorchLogicalNotOpTest, testInt64Default) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -10.0,
                   10.0);
}

TEST(cTorchLogicalNotOpTest, testUInt8Default) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, -10.0,
                   10.0);
}

TEST(cTorchLogicalNotOpTest, testBoolDefault) {
  test_logical_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_BOOL, -10.0, 10.0);
}
