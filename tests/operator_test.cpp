#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchOperatorTest, testForceInputOutputTsrNumEQ) {
  CTHOperator *op = create_dummy_op(CTH_OP_ID_abs, 1, 1);
  EXPECT_NO_FATAL_FAILURE(FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op));

  CTHOperator *op_2 = create_dummy_op(CTH_OP_ID_abs, 2, 1);
  EXPECT_EXIT(FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op_2), ::testing::ExitedWithCode(1),
              "Operator should have same numbers of input and output tensors.");
}

TEST(cTorchOperatorTest, testForceOpParamExist) {
  CTHOperator *op = create_dummy_op(CTH_OP_ID_abs, 1, 1);
  cth_tensor_dim_t dims[] = {20, 20};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHTensor *input = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  input->meta_info->tensor_name = "tensor_name";
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);

  const char *target_name = "tensor_name";
  EXPECT_NO_FATAL_FAILURE(
      FORCE_OP_INPUT_EXIST(op, target_name, CTH_TENSOR_DATA_TYPE_FLOAT_32));

  const char *target_name_2 = "tensor_name_2";
  EXPECT_EXIT(
      FORCE_OP_INPUT_EXIST(op, target_name_2, CTH_TENSOR_DATA_TYPE_FLOAT_32),
      ::testing::ExitedWithCode(1), "");
}

TEST(cTorchOperatorTest, testOpFailOnDtype) {
  CTHOperator *op = create_dummy_op(CTH_OP_ID_abs, 1, 1);
  cth_tensor_dim_t dims[] = {20, 20};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);

  CTHTensor *input = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);

  EXPECT_NO_FATAL_FAILURE(OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL));
  EXPECT_EXIT(OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_FLOAT_32),
              ::testing::ExitedWithCode(1),
              "Operator does not support data type.");
}

TEST(cTorchOperatorTest, testGetInputOutputByName) {
  CTHOperator *op = create_dummy_op(CTH_OP_ID_abs, 1, 1);
  cth_tensor_dim_t dims[] = {20, 20};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);

  const char *name = "name";
  CTHTensor *input = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  cth_tensor_set_name(input, name);
  CTHTensor *output = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  cth_tensor_set_name(output, name);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);
  cth_array_set(CTHTensor)(op->out_bound_tensors, 0, output);

  CTHTensor *result = cth_get_input_by_name(op, name, true);
  EXPECT_EQ(result, input);
  result = get_output_by_name(op, name, true);
  EXPECT_EQ(result, output);

  const char *name_2 = "name_2";
  cth_tensor_set_name(input, name_2);
  EXPECT_EXIT(cth_get_input_by_name(op, name, true),
              ::testing::ExitedWithCode(1), "Could not find tensor");
}

TEST(cTorchOperatorTest, testDeepFreeMEMRECORD) {
  cth_tensor_dim_t n_dim = 2;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 10;
  dims[1] = 20;
  CTHTensor *input = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  // another dims cause dims will be released in deep free. Here, avoid, double
  // free
  cth_tensor_dim_t *dims_2 =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims_2[0] = 10;
  dims_2[1] = 20;
  CTHTensor *output = create_dummy_tensor(
      dims_2, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  // param
  CTHParam *param = (CTHParam *)MALLOC(sizeof(CTHParam));
  param->type = CTH_PARAM_TYPE_MULTIPLIER;
  float multiplier = 0.5;
  param->data.multiplier = &multiplier;

  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_abs, 1, 1, 1);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);
  cth_array_set(CTHTensor)(op->out_bound_tensors, 0, output);
  cth_array_set(CTHParam)(op->params, 0, param);

  struct_deep_free(CTHOperator)(op);

  EXPECT_EQ(0, cth_get_num_unfree_records());
}

TEST(cTorchOperatorTest, testGetParam) {
  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_abs, 1, 1, 2);

  CTHParam *param_1 = (CTHParam *)MALLOC(sizeof(CTHParam));
  param_1->type = CTH_PARAM_TYPE_MULTIPLIER;
  float multiplier = 1.0;
  param_1->data.multiplier = &multiplier;
  cth_array_set(CTHParam)(op->params, 0, param_1);

  CTHParam *param_2 = (CTHParam *)MALLOC(sizeof(CTHParam));
  param_2->type = CTH_PARAM_TYPE_MIN;
  float multiplier_2 = -1.0;
  param_2->data.multiplier = &multiplier_2;
  cth_array_set(CTHParam)(op->params, 1, param_2);

  CTHParam *param = cth_get_param_by_type(op, CTH_PARAM_TYPE_MIN, true);
  EXPECT_EQ(param, param_2);

  param = cth_get_param_by_type(op, CTH_PARAM_TYPE_MULTIPLIER, true);
  EXPECT_EQ(param, param_1);
}
