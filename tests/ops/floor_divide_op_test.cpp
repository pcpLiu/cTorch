#include <tgmath.h>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_floor_divide(op, data_type, expect_fn)                         \
  do {                                                                         \
    CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);   \
    CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);   \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    data_type *input_1_ptr = (data_type *)input_1->values;                     \
    data_type *input_2_ptr = (data_type *)input_2->values;                     \
    data_type *output_ptr = (data_type *)output->values;                       \
    for (int i = 0; i < input_1->meta_info->n_elements; i++) {                 \
      data_type val = (data_type)floor(1.0 * input_1_ptr[i] / input_2_ptr[i]); \
      expect_fn(output_ptr[i], val);                                           \
    }                                                                          \
                                                                               \
  } while (0)

void test_floor_divide(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
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

  op_floor_divide_cpu(op);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_floor_divide(op, float, EXPECT_FLOAT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_floor_divide(op, double, EXPECT_DOUBLE_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_floor_divide(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_floor_divide(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_floor_divide(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_floor_divide(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_floor_divide(op, bool, EXPECT_EQ);
  }
}

TEST(cTorchFloorDivideOpTest, testFloat16Default) {
  test_floor_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0,
                    100.0);
}

TEST(cTorchFloorDivideOpTest, testFloat32Default) {
  test_floor_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0,
                    100.0);
}

TEST(cTorchFloorDivideOpTest, testFloat64Default) {
  test_floor_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0,
                    100.0);
}

TEST(cTorchFloorDivideOpTest, testInt16Default) {
  test_floor_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0,
                    100.0);
}

TEST(cTorchFloorDivideOpTest, testInt32Default) {
  test_floor_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0,
                    100.0);
}

TEST(cTorchFloorDivideOpTest, testInt64Default) {
  test_floor_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0,
                    100.0);
}

TEST(cTorchFloorDivideOpTest, testUInt8Default) {
  test_floor_divide(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0,
                    100.0);
}
