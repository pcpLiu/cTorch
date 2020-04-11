#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(operatorTest, testForceInputOutputTsrNumEQ) {
  CTorchOperator *op = create_dummy_op();
  tensor_dim dims[] = {20, 20};
  CTorchTensor *input =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  CTorchTensor *output =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  insert_list(CTorchTensor)(op->in_bound_tensors, input);
  insert_list(CTorchTensor)(op->out_bound_tensors, output);
  EXPECT_NO_FATAL_FAILURE(FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op));

  CTorchTensor *input2 =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  insert_list(CTorchTensor)(op->in_bound_tensors, input2);
  EXPECT_EXIT(FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op), ::testing::ExitedWithCode(1),
              "Operator should have same numbers of input and output tensors.");
}

TEST(operatorTest, testForceOpParamExist) {
  CTorchOperator *op = create_dummy_op();
  tensor_dim dims[] = {20, 20};
  CTorchTensor *input =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  input->meta_info->tensor_name = "tensor_name";
  const char *target_name = "tensor_name";
  insert_list(CTorchTensor)(op->in_bound_tensors, input);
  EXPECT_NO_FATAL_FAILURE(
      FORCE_OP_PARAM_EXIST(op, target_name, CTH_TENSOR_DATA_TYPE_FLOAT_32));

  const char *target_name_2 = "tensor_name_2";
  EXPECT_EXIT(
      FORCE_OP_PARAM_EXIST(op, target_name_2, CTH_TENSOR_DATA_TYPE_FLOAT_32),
      ::testing::ExitedWithCode(1), "FORCE_OP_PARAM_EXIST failes.");
}

TEST(operatorTest, testOpFailOnDtype) {
  CTorchOperator *op = create_dummy_op();
  tensor_dim dims[] = {20, 20};
  CTorchTensor *input =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  insert_list(CTorchTensor)(op->in_bound_tensors, input);
  EXPECT_NO_FATAL_FAILURE(OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL));
  EXPECT_EXIT(OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_FLOAT_32),
              ::testing::ExitedWithCode(1),
              "Operator does not support data type.");
}

TEST(operatorTest, testGetInputOutputByName) {
  CTorchOperator *op = create_dummy_op();
  tensor_dim dims[] = {20, 20};
  const char *name = "name";
  CTorchTensor *input =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  tensor_set_name(input, name);
  CTorchTensor *output =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);
  tensor_set_name(output, name);
  insert_list(CTorchTensor)(op->in_bound_tensors, input);
  insert_list(CTorchTensor)(op->out_bound_tensors, output);

  CTorchTensor *result = get_input_by_name(op, name, true);
  EXPECT_EQ(result, input);
  result = get_output_by_name(op, name, true);
  EXPECT_EQ(result, output);

  const char *name_2 = "name_2";
  tensor_set_name(input, name_2);
  EXPECT_EXIT(get_input_by_name(op, name, true), ::testing::ExitedWithCode(1),
              "Could not find tensor");
}