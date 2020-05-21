#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

TEST(cTorchMemUtilTest, testMalloc) {
  char *mem = (char *)MALLOC(200);
  MemoryRecord *record = cth_get_mem_record(mem);
  EXPECT_TRUE(record != nullptr);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_ALLOCATED, record->status);
}

TEST(cTorchMemUtilTest, testFree) {
  char *mem = (char *)MALLOC(200);
  MemoryRecord *record = cth_get_mem_record(mem);
  FREE(mem);
  EXPECT_TRUE(mem == nullptr);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record->status);

  // Free null
  mem = NULL;
  EXPECT_EXIT(FREE(mem), ::testing::ExitedWithCode(1),
              "Trying to free a NULL pointer.");
}
