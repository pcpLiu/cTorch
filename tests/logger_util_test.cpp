#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

TEST(cTorchLoggerUtilTest, testFailNullPtr) {
  char *ptr = NULL;
  EXPECT_EXIT(FAIL_NULL_PTR(ptr), ::testing::ExitedWithCode(1),
              "Pointer is NULL.");
}

TEST(cTorchLoggerUtilTest, testFailExit) {
  EXPECT_EXIT(FAIL_EXIT(CTH_LOG_ERR, "Fail exit test %s", "blackhole"),
              ::testing::ExitedWithCode(1), "Fail exit test blackhole");
}
