#include "ctorch/c_torch.h"
#include "gtest/gtest.h"

TEST(CommonUtilTest, testFailExit) {
  EXPECT_EXIT(FAIL_EXIT(CTH_LOG_STR, "Test out"), ::testing::ExitedWithCode(1),
              "Test out");
}