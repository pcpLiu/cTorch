#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchStorageTest, testTensorNameMatch) {
  cth_tensor_dim_t dims[] = {10, 20};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  char name[] = "Test_name";
  CTHTensor *tensor =
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);
  tensor->meta_info->tensor_name = name;
  EXPECT_TRUE(cth_tensor_name_match(tensor, name));
}

TEST(cTorchStorageTest, testForceTensorDimension) {
  cth_tensor_dim_t dims[] = {10, 20};
  cth_tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTHTensor *tensor =
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);

  cth_tensor_dim_t test_dims[] = {30, 20};
  EXPECT_NO_FATAL_FAILURE(CTH_FORCE_TENSOR_DIMENSION(tensor, dims, 2));
  EXPECT_EXIT(CTH_FORCE_TENSOR_DIMENSION(tensor, test_dims, 2),
              ::testing::ExitedWithCode(1),
              "CTH_FORCE_TENSOR_DIMENSION failes.");
}

TEST(cTorchStorageTest, testFreeTensor) {
  cth_tensor_dim_t n_dim = 2;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 10;
  dims[1] = 20;
  CTHTensor *tensor =
      create_dummy_tensor(dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 2.0);
  CTHMemoryRecord *record_tensor = cth_get_mem_record(tensor);
  CTHMemoryRecord *record_meta = cth_get_mem_record(tensor->meta_info);
  CTHMemoryRecord *record_name =
      cth_get_mem_record(tensor->meta_info->tensor_name);
  CTHMemoryRecord *record_dims = cth_get_mem_record(tensor->meta_info->dims);
  CTHMemoryRecord *record_values = cth_get_mem_record(tensor->values);

  struct_deep_free(CTHTensor)(tensor);

  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_tensor->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_meta->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_name->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_dims->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_values->status);
}

TEST(cTorchStorageTest, testReductionStartoffset) {
  cth_tensor_dim_t n_dim = 4;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 2;
  dims[3] = 4;
  CTHTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  // reduce dim 1
  cth_tensor_dim_t reduce_dim = 1;

  cth_tensor_dim_t index_dims[] = {0, 0, 0};
  cth_tensor_dim_t startoffset =
      cth_tensor_reduce_startoffset(tensor, index_dims, reduce_dim);
  EXPECT_EQ(startoffset, 0);

  cth_tensor_dim_t index_dims_1[] = {0, 1, 0};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_1, reduce_dim);
  EXPECT_EQ(startoffset, 4);

  cth_tensor_dim_t index_dims_2[] = {1, 0, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_2, reduce_dim);
  EXPECT_EQ(startoffset, 27);

  cth_tensor_dim_t index_dims_3[] = {1, 1, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_3, reduce_dim);
  EXPECT_EQ(startoffset, 31);

  cth_tensor_dim_t index_dims_4[] = {0, 1, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_4, reduce_dim);
  EXPECT_EQ(startoffset, 7);

  // reduce dim 0
  reduce_dim = 0;

  cth_tensor_dim_t index_dims_5[] = {1, 0, 3};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_5, reduce_dim);
  EXPECT_EQ(startoffset, 11);

  cth_tensor_dim_t index_dims_6[] = {0, 0, 1};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_6, reduce_dim);
  EXPECT_EQ(startoffset, 1);

  // reduce dim 3
  reduce_dim = 3;

  cth_tensor_dim_t index_dims_7[] = {1, 2, 1};
  startoffset = cth_tensor_reduce_startoffset(tensor, index_dims_7, reduce_dim);
  EXPECT_EQ(startoffset, 44);
}

TEST(cTorchStorageTest, testReductionInnerOffset) {
  cth_tensor_dim_t n_dim = 4;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 2;
  dims[3] = 4;
  CTHTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  // reduce dime 1
  cth_tensor_dim_t reduce_dim = 1;
  cth_tensor_dim_t innerOffset =
      cth_tensor_reduce_inneroffset(tensor, reduce_dim);
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
  cth_tensor_dim_t n_dim = 4;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 3;
  dims[1] = 1;
  dims[2] = 3;
  dims[3] = 2;
  CTHTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  cth_tensor_dim_t reduce_dim = 3;

  cth_tensor_dim_t reduce_index[] = {0, 0, 0};
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

TEST(cTorchStorageTest, testTensorAfterDimOffset) {
  cth_tensor_dim_t n_dim = 4;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 3;
  dims[1] = 1;
  dims[2] = 5;
  dims[3] = 2;
  CTHTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  cth_tensor_dim_t offset = 0;

  offset = cth_tensor_after_dim_offset(tensor, 0);
  EXPECT_EQ(offset, dims[1] * dims[2] * dims[3]);

  offset = cth_tensor_after_dim_offset(tensor, 1);
  EXPECT_EQ(offset, dims[2] * dims[3]);

  offset = cth_tensor_after_dim_offset(tensor, 2);
  EXPECT_EQ(offset, dims[3]);

  offset = cth_tensor_after_dim_offset(tensor, 3);
  EXPECT_EQ(offset, 1);
}

TEST(cTorchStorageTest, testTensorAt) {
  cth_tensor_dim_t n_dim = 4;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 3;
  dims[1] = 6;
  dims[2] = 5;
  dims[3] = 7;
  CTHTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  float *val = (float *)MALLOC(sizeof(float));
  *val = 0.0;
  cth_tensor_dim_t offset = 0;

  cth_tensor_at(tensor, val, 2, 3, 3, 4);
  offset = 2 * cth_tensor_after_dim_offset(tensor, 0) +
           3 * cth_tensor_after_dim_offset(tensor, 1) +
           3 * cth_tensor_after_dim_offset(tensor, 2) +
           4 * cth_tensor_after_dim_offset(tensor, 3);
  EXPECT_EQ(((float *)tensor->values)[offset], *val);

  cth_tensor_at(tensor, val, 0, 1, 0, 4);
  offset = 0 * cth_tensor_after_dim_offset(tensor, 0) +
           1 * cth_tensor_after_dim_offset(tensor, 1) +
           0 * cth_tensor_after_dim_offset(tensor, 2) +
           4 * cth_tensor_after_dim_offset(tensor, 3);
  EXPECT_EQ(((float *)tensor->values)[offset], *val);

  cth_tensor_at(tensor, val, 0, 0, 0, 0);
  offset = 0 * cth_tensor_after_dim_offset(tensor, 0) +
           0 * cth_tensor_after_dim_offset(tensor, 1) +
           0 * cth_tensor_after_dim_offset(tensor, 2) +
           0 * cth_tensor_after_dim_offset(tensor, 3);
  EXPECT_EQ(((float *)tensor->values)[offset], *val);

  // out of boundary
  EXPECT_EXIT(cth_tensor_at(tensor, val, 1, 2, 1000, 3),
              ::testing::ExitedWithCode(1), "out of boundary");
}
