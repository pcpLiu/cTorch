#include "gtest/gtest.h"

int main(int argc, char **argv) {
  srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  RUN_ALL_TESTS();
  // If tests failed, don't make it fail gcov
  return 0;
}