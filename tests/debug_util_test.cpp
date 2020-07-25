#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

TEST(cTorchDebugUtilTest, testGetRecord) {
  CTHTensor *tensor = (CTHTensor *)malloc(sizeof(CTHTensor));
  CTHMemoryRecord *record = cth_add_mem_record(tensor);
  EXPECT_EQ(tensor, record->addr);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_ALLOCATED, record->status);

  CTHMemoryRecord *record_dup = cth_add_mem_record(tensor);
  EXPECT_EQ(record_dup, record);
}

TEST(cTorchDebugUtilTest, testAddRecord) {
  CTHTensor *tensor = (CTHTensor *)malloc(sizeof(CTHTensor));
  CTHMemoryRecord *record = cth_add_mem_record(tensor);
  CTHMemoryRecord *record_fetch = cth_get_mem_record(tensor);
  EXPECT_EQ(record, record_fetch);

  CTHTensor *tensor_2 = (CTHTensor *)malloc(sizeof(CTHTensor));
  CTHMemoryRecord *record_fetch_2 = cth_get_mem_record(tensor_2);
  EXPECT_EQ(record_fetch_2, nullptr);
}
