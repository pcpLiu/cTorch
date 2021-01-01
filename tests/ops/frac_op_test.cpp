#include <ctgmath>

#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#define _verify_frac(op, data_type, expect_fn)                                 \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    data_type *input_t = (data_type *)input->values;                           \
    data_type *output_t = (data_type *)output->values;                         \
    double holder = 0;                                                         \
    for (cth_tensor_dim_t i = 0; i < input->meta_info->n_elements; i++) {      \
      data_type val = (data_type)modf(input_t[i], &holder);                    \
      expect_fn(output_t[i], val);                                             \
    }                                                                          \
                                                                               \
  } while (0)
void test_frac(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min,
               float max) {
  cth_tensor_dim_t dims[] = {100, 100};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_frac, dims, n_dim,
                                                data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_frac_cpu(op);
  } else if (backend == CTH_BACKEND_MKL) {
#ifdef BACKEND_MKL
    op_frac_mkl(op);
#endif
  }

  sample_print(data_type,
               cth_array_at(CTHTensor)(op->in_bound_tensors, 0)->values,
               cth_array_at(CTHTensor)(op->out_bound_tensors, 0)->values, 2);

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _verify_frac(op, float, EXPECT_FLOAT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _verify_frac(op, double, EXPECT_DOUBLE_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _verify_frac(op, int16_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _verify_frac(op, int32_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _verify_frac(op, int64_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _verify_frac(op, uint8_t, EXPECT_EQ);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    _verify_frac(op, bool, EXPECT_EQ);
  }
}

TEST(cTorchFracOpTest, testFloat16Default) {
  test_frac(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, -10.0, 10.0);
}

TEST(cTorchFracOpTest, testFloat32Default) {
  test_frac(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
}

TEST(cTorchFracOpTest, testFloat64Default) {
  test_frac(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
}

#ifdef BACKEND_MKL
TEST(cTorchFracOpTest, testFloat32MKL) {
  test_frac(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_32, -10.0, 10.0);
}

TEST(cTorchFracOpTest, testFloat64MKL) {
  test_frac(CTH_BACKEND_MKL, CTH_TENSOR_DATA_TYPE_FLOAT_64, -10.0, 10.0);
}
#endif

TEST(cTorchFracOpTest, testInt16Default) {
  test_frac(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, -10.0, 10.0);
}

TEST(cTorchFracOpTest, testInt32Default) {
  test_frac(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, -10.0, 10.0);
}

TEST(cTorchFracOpTest, testInt64Default) {
  test_frac(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, -10.0, 10.0);
}

TEST(cTorchFracOpTest, testUInt8Default) {
  test_frac(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 0.0, 10.0);
}
