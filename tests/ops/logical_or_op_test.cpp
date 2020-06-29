#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_logical_or(op, data_type, expect_fn)                           \
  do {                                                                         \
    CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *input_1_t = (data_type *)input_1->values;                       \
    data_type *input_2_t = (data_type *)input_2->values;                       \
    bool *output_t = (bool *)output->values;                                   \
                                                                               \
    for (int i = 0; i < input_1->meta_info->n_elements; i++) {                 \
      uint16_t x = (0 == input_1_t[i] ? 0 : 1);                                \
      uint16_t y = (0 == input_2_t[i] ? 0 : 1);                                \
      bool val;                                                                \
      if (x == 1 || y == 1) {                                                  \
        val = true;                                                            \
      } else {                                                                 \
        val = false;                                                           \
      }                                                                        \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
  } while (0)

void test_logical_or(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
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
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_BOOL, min, max));

  op_logical_or_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_logical_or(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_logical_or(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_logical_or(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_logical_or(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_logical_or(op, bool, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
             data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_logical_or(op, float, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_logical_or(op, double, EXPECT_EQ);
  }
}

TEST(cTorchLogicalOrOpTest, testFloat16Default) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -10.0,
                  10.0);
}

TEST(cTorchLogicalOrOpTest, testFloat32Default) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0,
                  10.0);
}

TEST(cTorchLogicalOrOpTest, testFloat64Default) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0,
                  10.0);
}

TEST(cTorchLogicalOrOpTest, testInt16Default) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0,
                  10.0);
}

TEST(cTorchLogicalOrOpTest, testInt32Default) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -10.0,
                  10.0);
}

TEST(cTorchLogicalOrOpTest, testInt64Default) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -10.0,
                  10.0);
}

TEST(cTorchLogicalOrOpTest, testUInt8Default) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, -10.0,
                  10.0);
}

TEST(cTorchLogicalOrOpTest, testBoolDefault) {
  test_logical_or(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_BOOL, -10.0, 10.0);
}
