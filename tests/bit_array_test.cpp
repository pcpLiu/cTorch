#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

#include <tgmath.h>

TEST(cTorchBitsArrayTest, testCreate) {
  bit_array_t *bits_array = cth_new_bit_array(200);
  EXPECT_EQ(bits_array->num_ints, ceil(200 / 32));
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