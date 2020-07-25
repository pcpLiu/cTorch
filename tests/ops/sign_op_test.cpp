#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"
#include <tgmath.h>

#define _verify_sign(op, data_type, expect_fn)                                 \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *input_t = (data_type *)input->values;                           \
    data_type *output_t = (data_type *)output->values;                         \
    for (cth_tensor_dim_t i = 0; i < input->meta_info->n_elements; i++) {      \
      data_type val = 0;                                                       \
      if (input_t[i] != 0) {                                                   \
        val = (data_type)(input_t[i] > 0 ? 1 : -1);                            \
      }                                                                        \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
                                                                               \
  } while (0)

void test_sign(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_abs, dims, n_dim,
                                                data_type, min, max);
  CTHOperator *op = op_node->conent.op;
  op_sign_cpu(op);

  sample_print(data_type,
               cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->values,
               cth_array_at(CTHTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_sign(op, float, EXPECT_FLOAT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_sign(op, double, EXPECT_DOUBLE_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_sign(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_sign(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_sign(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_sign(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_sign(op, bool, EXPECT_EQ);
  }
}

TEST(cTorchSignOpTest, testFloat16Default) {
  test_sign(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -10.0, 10.0);
}

TEST(cTorchSignOpTest, testFloat32Default) {
  test_sign(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
}

TEST(cTorchSignOpTest, testFloat64Default) {
  test_sign(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
}

TEST(cTorchSignOpTest, testInt16Default) {
  test_sign(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0, 10.0);
}

TEST(cTorchSignOpTest, testInt32Default) {
  test_sign(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -10.0, 10.0);
}

TEST(cTorchSignOpTest, testInt64Default) {
  test_sign(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -10.0, 10.0);
}

TEST(cTorchSignOpTest, testUInt8Default) {
  test_sign(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 10.0);
}
