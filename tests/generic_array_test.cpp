#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

def_array(int);
impl_new_array_func(int);
impl_array_at_func(int);
impl_array_set_func(int);
impl_free_array_deep_func(int);

TEST(cTorchArrayTest, testCreate) {
  Array(int) *array = new_array(int)(100);
  EXPECT_EQ(array->size, 100);
  for (array_index_t i = 0; i < 100; i++) {
    EXPECT_EQ(*(array->_data + i), nullptr);
  }
}

TEST(cTorchArrayTest, testSet) {
  Array(int) *array = new_array(int)(4);

  int *val_1 = heap_int(1);
  int *val_2 = heap_int(2);
  int *val_3 = heap_int(3);
  int *val_4 = heap_int(4);

  array_set(int)(array, 0, val_1);
  array_set(int)(array, 1, val_2);
  array_set(int)(array, 2, val_3);
  array_set(int)(array, 3, val_4);

  EXPECT_EQ(*(array->_data + 0), val_1);
  EXPECT_EQ(*(array->_data + 1), val_2);
  EXPECT_EQ(*(array->_data + 2), val_3);
  EXPECT_EQ(*(array->_data + 3), val_4);
}

TEST(cTorchArrayTest, testAt) {
  Array(int) *array = new_array(int)(4);

  int *val_1 = heap_int(1);
  int *val_2 = heap_int(2);
  int *val_3 = heap_int(3);
  int *val_4 = heap_int(4);

  array_set(int)(array, 0, val_1);
  array_set(int)(array, 1, val_2);
  array_set(int)(array, 2, val_3);
  array_set(int)(array, 3, val_4);

  EXPECT_EQ(array_at(int)(array, 0), val_1);
  EXPECT_EQ(array_at(int)(array, 1), val_2);
  EXPECT_EQ(array_at(int)(array, 2), val_3);
  EXPECT_EQ(array_at(int)(array, 3), val_4);
}

TEST(cTorchArrayTest, testErr) {
  Array(int) *array = new_array(int)(4);
  int *val_1 = heap_int(1);

  EXPECT_EXIT(array_set(int)(array, 10, val_1), ::testing::ExitedWithCode(1),
              "");
  EXPECT_EXIT(array_at(int)(array, 10), ::testing::ExitedWithCode(1), "");
  EXPECT_EXIT(array_set(int)(NULL, 10, val_1), ::testing::ExitedWithCode(1),
              "");
}

TEST(cTorchArrayTest, testDeepFreeArray) {
  Array(int) *array = new_array(int)(3);
  MemoryRecord *record_array = cth_get_mem_record(array);

  int *val_1 = heap_int(1);
  MemoryRecord *record_data_1 = cth_get_mem_record(val_1);
  array_set(int)(array, 0, val_1);

  int *val_2 = heap_int(2);
  MemoryRecord *record_data_2 = cth_get_mem_record(val_2);
  array_set(int)(array, 1, val_2);

  int *val_3 = heap_int(3);
  MemoryRecord *record_data_3 = cth_get_mem_record(val_3);
  array_set(int)(array, 2, val_3);

  free_array_deep(int)(array);

  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_array->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_data_1->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_data_2->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_data_3->status);
}
