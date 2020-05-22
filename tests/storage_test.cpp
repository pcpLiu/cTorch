#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchStorageTest, testTensorNameMatch) {
  tensor_dim_t dims[] = {10, 20};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  char name[] = "Test_name";
  CTorchTensor *tensor =
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);
  tensor->meta_info->tensor_name = name;
  EXPECT_TRUE(cth_tensor_name_match(tensor, name));
}

TEST(cTorchStorageTest, testForceTensorDimension) {
  tensor_dim_t dims[] = {10, 20};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchTensor *tensor =
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);

  tensor_dim_t test_dims[] = {30, 20};
  EXPECT_NO_FATAL_FAILURE(FORCE_TENSOR_DIMENSION(tensor, dims));
  EXPECT_EXIT(FORCE_TENSOR_DIMENSION(tensor, test_dims),
              ::testing::ExitedWithCode(1), "FORCE_TENSOR_DIMENSION failes.");
}

TEST(cTorchStorageTest, testFreeTensor) {
  tensor_dim_t n_dim = 2;
  tensor_dim_t *dims = (tensor_dim_t *)MALLOC(n_dim * sizeof(tensor_dim_t));
  dims[0] = 10;
  dims[1] = 20;
  CTorchTensor *tensor =
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);
  MemoryRecord *record_tensor = cth_get_mem_record(tensor);
  MemoryRecord *record_meta = cth_get_mem_record(tensor->meta_info);
  MemoryRecord *record_name =
      cth_get_mem_record(tensor->meta_info->tensor_name);
  MemoryRecord *record_dims = cth_get_mem_record(tensor->meta_info->dims);
  MemoryRecord *record_values = cth_get_mem_record(tensor->values);

  struct_deep_free(CTorchTensor)(tensor);

  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_tensor->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_meta->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_name->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_dims->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_values->status);
}