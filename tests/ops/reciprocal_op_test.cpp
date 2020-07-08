#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _cth_reciprocal_test(x) (1 / x);

void test_reciprocal(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
                     float min, float max) {
  tensor_dim_t dims[] = {100, 100};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_neg, dims, n_dim,
                                                   data_type, min, max);
  CTorchOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_reciprocal_cpu(op);
  } else if (backend == CTH_BACKEND_APPLE) {
#ifdef BACKEND_APPLE
    op_reciprocal_apple(op);
#endif
  }

  sample_print(data_type,
               array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
               array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_unary(op, float, EXPECT_FLOAT_EQ, _cth_reciprocal_test);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_unary(op, double, EXPECT_DOUBLE_EQ, _cth_reciprocal_test);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_unary(op, int16_t, EXPECT_EQ, _cth_reciprocal_test);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_unary(op, int32_t, EXPECT_EQ, _cth_reciprocal_test);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_unary(op, int64_t, EXPECT_EQ, _cth_reciprocal_test);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_unary(op, uint8_t, EXPECT_EQ, _cth_reciprocal_test);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _ele_wise_equal_unary(op, bool, EXPECT_EQ, _cth_reciprocal_test);
  }
}

TEST(cTorchReciprocalOpTest, testFloat16Default) {
  test_reciprocal(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0,
                  100.0);
}

TEST(cTorchReciprocalOpTest, testFloat32Default) {
  test_reciprocal(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0,
                  100.0);
}

TEST(cTorchReciprocalOpTest, testFloat64Default) {
  test_reciprocal(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0,
                  100.0);
}

#ifdef BACKEND_APPLE
TEST(cTorchReciprocalOpTest, testFloat64Apple) {
  test_reciprocal(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchReciprocalOpTest, testFloat32Apple) {
  test_reciprocal(CTH_BACKEND_APPLE, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}
#endif

TEST(cTorchReciprocalOpTest, testInt16Default) {
  test_reciprocal(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchReciprocalOpTest, testInt32Default) {
  test_reciprocal(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchReciprocalOpTest, testInt64Default) {
  test_reciprocal(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}

TEST(cTorchReciprocalOpTest, testUInt8Default) {
  test_reciprocal(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 100.0);
}
