#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

#include <tgmath.h>

TEST(cTorchBitsArrayTest, testCreate) {
  bit_array_t *bits_array = cth_new_bit_array(200);
  EXPECT_EQ(bits_array->num_ints, ceil((double)200 / (double)32));
  EXPECT_EQ(bits_array->size, 200);
}

TEST(cTorchBitsArrayTest, testSet) {
  bit_array_t *bits_array = cth_new_bit_array(200);
  cth_set_bit(bits_array, 0);
  cth_set_bit(bits_array, 17);
  cth_set_bit(bits_array, 100);
  cth_set_bit(bits_array, 199);

  uint32_t flag = 1;
  flag = flag << 0;
  EXPECT_TRUE((*(bits_array->bits + 0 / 32) & flag) != 0);

  flag = 1;
  flag = flag << (17 % 32);
  EXPECT_TRUE((*(bits_array->bits + 17 / 32) & flag) != 0);

  flag = 1;
  flag = flag << (100 % 32);
  EXPECT_TRUE((*(bits_array->bits + 100 / 32) & flag) != 0);

  flag = 1;
  flag = flag << (199 % 32);
  EXPECT_TRUE((*(bits_array->bits + 199 / 32) & flag) != 0);
}

TEST(cTorchBitsArrayTest, testClear) {
  bit_array_t *bits_array = cth_new_bit_array(200);
  cth_set_bit(bits_array, 0);
  cth_set_bit(bits_array, 17);
  cth_set_bit(bits_array, 100);
  cth_set_bit(bits_array, 199);

  cth_clear_bit(bits_array, 0);
  uint32_t flag = 1;
  flag = flag << 0;
  EXPECT_EQ((*(bits_array->bits + 0 / 32) & flag), 0);

  cth_clear_bit(bits_array, 17);
  flag = 1;
  flag = flag << (17 % 32);
  EXPECT_EQ((*(bits_array->bits + 17 / 32) & flag), 0);

  cth_clear_bit(bits_array, 100);
  flag = 1;
  flag = flag << (100 % 32);
  EXPECT_EQ((*(bits_array->bits + 100 / 32) & flag), 0);

  cth_clear_bit(bits_array, 199);
  flag = 1;
  flag = flag << (199 % 32);
  EXPECT_EQ((*(bits_array->bits + 199 / 32) & flag), 0);
}

TEST(cTorchBitsArrayTest, testCheck) {
  bit_array_t *bits_array = cth_new_bit_array(200);
  cth_set_bit(bits_array, 0);
  cth_set_bit(bits_array, 17);
  cth_set_bit(bits_array, 100);
  cth_set_bit(bits_array, 199);

  EXPECT_EQ(cth_is_bit_set(bits_array, 0), true);
  EXPECT_EQ(cth_is_bit_set(bits_array, 17), true);
  EXPECT_EQ(cth_is_bit_set(bits_array, 100), true);
  EXPECT_EQ(cth_is_bit_set(bits_array, 199), true);

  EXPECT_EQ(cth_is_bit_set(bits_array, 99), false);
  EXPECT_EQ(cth_is_bit_set(bits_array, 33), false);
}

TEST(cTorchBitsArrayTest, testAllClearCheck) {
  bit_array_t *bits_array = cth_new_bit_array(500);
  cth_set_bit(bits_array, 0);
  cth_set_bit(bits_array, 17);
  cth_set_bit(bits_array, 100);
  cth_set_bit(bits_array, 199);

  EXPECT_EQ(cth_are_all_bits_set(bits_array), false);
  EXPECT_EQ(cth_are_all_bits_clear(bits_array), false);

  for (bit_array_index_t i = 0; i < 500; i++) {
    cth_set_bit(bits_array, i);
  }
  EXPECT_EQ(cth_are_all_bits_set(bits_array), true);
  EXPECT_EQ(cth_are_all_bits_clear(bits_array), false);

  bit_array_t *bits_array_2 = cth_new_bit_array(4);
  cth_set_bit(bits_array_2, 0);
  cth_set_bit(bits_array_2, 1);
  cth_set_bit(bits_array_2, 2);
  cth_set_bit(bits_array_2, 3);

  EXPECT_EQ(cth_are_all_bits_set(bits_array_2), true);
  EXPECT_EQ(cth_are_all_bits_clear(bits_array_2), false);
}