#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

namespace cTorchArrayTest {
// for testFreeListDeep
int *heap_int(int x) {
  int *ptr = (int *)MALLOC(sizeof(int));
  *ptr = x;
  return ptr;
}

// for testFreeListDeep
void free_deep_int(int *x) { FREE(x); }
} // namespace cTorchArrayTest

def_array(int);
impl_new_array_func(int);
impl_array_at_func(int);
impl_array_set_func(int);

TEST(cTorchArrayTest, testCreate) {
  Array(int) *array = new_array(int)(100);
  EXPECT_EQ(array->size, 100);
  for (array_index_t i = 0; i < 100; i++) {
    EXPECT_EQ(*(array->_data + i), nullptr);
  }
}

TEST(cTorchArrayTest, testSet) {
  Array(int) *array = new_array(int)(4);

  int *val_1 = cTorchArrayTest::heap_int(1);
  int *val_2 = cTorchArrayTest::heap_int(2);
  int *val_3 = cTorchArrayTest::heap_int(3);
  int *val_4 = cTorchArrayTest::heap_int(4);

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

  int *val_1 = cTorchArrayTest::heap_int(1);
  int *val_2 = cTorchArrayTest::heap_int(2);
  int *val_3 = cTorchArrayTest::heap_int(3);
  int *val_4 = cTorchArrayTest::heap_int(4);

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
  int *val_1 = cTorchArrayTest::heap_int(1);

  EXPECT_EXIT(array_set(int)(array, 10, val_1), ::testing::ExitedWithCode(1),
              "");
  EXPECT_EXIT(array_at(int)(array, 10), ::testing::ExitedWithCode(1), "");
  EXPECT_EXIT(array_set(int)(NULL, 10, val_1), ::testing::ExitedWithCode(1),
              "");
}