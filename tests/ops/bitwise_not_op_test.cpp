#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_bitwise_not(op, data_type, expect_fn)                          \
  do {                                                                         \
    CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);     \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *input_t = (data_type *)input->values;                           \
    data_type *output_t = (data_type *)output->values;                         \
                                                                               \
    for (int i = 0; i < input->meta_info->n_elements; i++) {                   \
      data_type val = ~input_t[i];                                             \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
  } while (0)

void test_bitwise_not(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                      float min, float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_abs, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;

  op_bitwise_not_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_bitwise_not(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_bitwise_not(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_bitwise_not(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_bitwise_not(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_bitwise_not(op, bool, EXPECT_EQ);
  }
}

TEST(cTorchBitwiseNotOpTest, testInt16Default) {
  test_bitwise_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0,
                   100.0);
}

TEST(cTorchBitwiseNotOpTest, testInt32Default) {
  test_bitwise_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0,
                   100.0);
}

TEST(cTorchBitwiseNotOpTest, testInt64Default) {
  test_bitwise_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0,
                   100.0);
}

TEST(cTorchBitwiseNotOpTest, testUInt8Default) {
  test_bitwise_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0,
                   100.0);
}

TEST(cTorchBitwiseNotOpTest, testBoolDefault) {
  test_bitwise_not(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_BOOL, 1.0, 100.0);
}