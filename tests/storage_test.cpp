#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(storageTest, testTensorNameMatch) {
  tensor_dim dims[] = {10, 20};
  char name[] = "Test_name";
  CTorchTensor *tensor =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);
  tensor->meta_info->tensor_name = name;
  EXPECT_TRUE(tensor_name_match(tensor, name));
}

TEST(storageTest, testForceTensorDimension) {
  tensor_dim dims[] = {10, 20};
  CTorchTensor *tensor =
      create_dummy_tensor(dims, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);

  tensor_dim test_dims[] = {30, 20};
  EXPECT_NO_FATAL_FAILURE(FORCE_TENSOR_DIMENSION(tensor, dims));
  EXPECT_EXIT(FORCE_TENSOR_DIMENSION(tensor, test_dims),
              ::testing::ExitedWithCode(1), "FORCE_TENSOR_DIMENSION failes.");
}