#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

void test_acos(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  uint32_t dims[] = {100, 100};
  CTorchNode *op_node =
      create_dummy_op_node(CTH_OP_ID_acos, dims, data_type, min, max);
  execute_node(op_node, backend);
  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal(float, EXPECT_FLOAT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal(double, EXPECT_DOUBLE_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal(int16_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal(int32_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal(int64_t, EXPECT_EQ, acos);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal(uint8_t, EXPECT_EQ, acos);
  }
}

TEST(acosOpTest, testFloat16X86) {
  test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_FLOAT_16, -1.0, 1.0);
}

TEST(acosOpTest, testFloat32X86) {
  test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
}

TEST(acosOpTest, testFloat64X86) {
  test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_FLOAT_64, -1.0, 1.0);
}

TEST(acosOpTest, testInt16X86) {
  test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_INT_16, -1.0, 1.0);
}

TEST(acosOpTest, testInt32X86) {
  test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_INT_32, -1.0, 1.0);
}

TEST(acosOpTest, testInt64X86) {
  test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_INT_64, -1.0, 1.0);
}

TEST(acosOpTest, testUInt8X86) {
  test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 1.0);
}

TEST(acosOpTest, testBoolX86ExpectExit) {
  CTH_NAN_EXIT = true;
  uint32_t dims[] = {1, 1};
  CTorchNode *op_node = create_dummy_op_node(CTH_OP_ID_acos, dims,
                                             CTH_TENSOR_DATA_TYPE_BOOL, 0, 0);
  EXPECT_EXIT(execute_node(op_node, CTH_BACKEND_CPU_X86),
              ::testing::ExitedWithCode(1),
              "Operator does not support data type");
}

TEST(acosOpTest, testInvalidInputExit) {
  CTH_NAN_EXIT = true;
  EXPECT_EXIT(
      test_acos(CTH_BACKEND_CPU_X86, CTH_TENSOR_DATA_TYPE_FLOAT_16, 100, 200),
      ::testing::ExitedWithCode(1), "Value is NaN");
}

TEST(acosOpTest, testInvalidInputKeep) {
  CTH_NAN_EXIT = false;
  uint32_t dims[] = {1, 1};
  CTorchNode *op_node = create_dummy_op_node(
      CTH_OP_ID_acos, dims, CTH_TENSOR_DATA_TYPE_FLOAT_16, 100, 200);
  execute_node(op_node, CTH_BACKEND_CPU_X86);
  EXPECT_TRUE(
      tensor_all_nan(op_node->conent.op->out_bound_tensors->head->data));
}