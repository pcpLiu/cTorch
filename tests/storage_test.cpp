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
  EXPECT_NO_FATAL_FAILURE(FORCE_TENSOR_DIMENSION(tensor, dims, 2));
  EXPECT_EXIT(FORCE_TENSOR_DIMENSION(tensor, test_dims, 2),
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

TEST(cTorchStorageTest, testReductionStartoffset) {
  tensor_dim_t n_dim = 4;
  tensor_dim_t *dims = (tensor_dim_t *)MALLOC(n_dim * sizeof(tensor_dim_t));
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 2;
  dims[3] = 4;
  CTorchTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  // reduce dim 1
  tensor_dim_t reduce_dim = 1;

  tensor_dim_t index_dims[] = {0, 0, 0};
  tensor_dim_t startoffset =
      cth_tensor_reduce_startoffset(tensor, index_dims, reduce_dim);
  EXPECT_EQ(startoffset, 0);

  tensor_dim_t index_dims_1[] = {0, 1, 0};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_1, reduce_dim);
  EXPECT_EQ(startoffset, 4);

  tensor_dim_t index_dims_2[] = {1, 0, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_2, reduce_dim);
  EXPECT_EQ(startoffset, 27);

  tensor_dim_t index_dims_3[] = {1, 1, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_3, reduce_dim);
  EXPECT_EQ(startoffset, 31);

  tensor_dim_t index_dims_4[] = {0, 1, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_4, reduce_dim);
  EXPECT_EQ(startoffset, 7);

  // reduce dim 0
  reduce_dim = 0;

  tensor_dim_t index_dims_5[] = {1, 0, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_5, reduce_dim);
  EXPECT_EQ(startoffset, 11);

  tensor_dim_t index_dims_6[] = {0, 0, 1};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_6, reduce_dim);
  EXPECT_EQ(startoffset, 1);

  // reduce dim 3
  reduce_dim = 3;

  tensor_dim_t index_dims_7[] = {1, 2, 1};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_7, reduce_dim);
  EXPECT_EQ(startoffset, 44);
}

TEST(cTorchStorageTest, testReductionInnerOffset) {
  tensor_dim_t n_dim = 4;
  tensor_dim_t *dims = (tensor_dim_t *)MALLOC(n_dim * sizeof(tensor_dim_t));
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 2;
  dims[3] = 4;
  CTorchTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  // reduce dime 1
  tensor_dim_t reduce_dim = 1;
  tensor_dim_t innerOffset = cth_tensor_reduce_inneroffset(tensor, reduce_dim);
  EXPECT_EQ(innerOffset, 8);

  // reduce dim 0
  reduce_dim = 0;
  innerOffset = cth_tensor_reduce_inneroffset(tensor, reduce_dim);
  EXPECT_EQ(innerOffset, 24);

  // reduce dim 0
  reduce_dim = 3;
  innerOffset = cth_tensor_reduce_inneroffset(tensor, reduce_dim);
  EXPECT_EQ(innerOffset, 1);
}

TEST(cTorchStorageTest, testReductionIndexGeneration) {
  tensor_dim_t n_dim = 4;
  tensor_dim_t *dims = (tensor_dim_t *)MALLOC(n_dim * sizeof(tensor_dim_t));
  dims[0] = 3;
  dims[1] = 1;
  dims[2] = 3;
  dims[3] = 2;
  CTorchTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  tensor_dim_t reduce_dim = 3;

  tensor_dim_t reduce_index[] = {0, 0, 0};
  cth_tensor_get_reduce_index(tensor, 1, reduce_dim, reduce_index);
  EXPECT_EQ(0, reduce_index[0]);
  EXPECT_EQ(0, reduce_index[1]);
  EXPECT_EQ(1, reduce_index[2]);

  reduce_dim = 1;

  cth_tensor_get_reduce_index(tensor, 1, reduce_dim, reduce_index);
  EXPECT_EQ(0, reduce_index[0]);
  EXPECT_EQ(0, reduce_index[1]);
  EXPECT_EQ(1, reduce_index[2]);

  cth_tensor_get_reduce_index(tensor, 5, reduce_dim, reduce_index);
  EXPECT_EQ(0, reduce_index[0]);
  EXPECT_EQ(2, reduce_index[1]);
  EXPECT_EQ(1, reduce_index[2]);

  reduce_dim = 0;

  cth_tensor_get_reduce_index(tensor, 3, reduce_dim, reduce_index);
  EXPECT_EQ(0, reduce_index[0]);
  EXPECT_EQ(1, reduce_index[1]);
  EXPECT_EQ(1, reduce_index[2]);
}
