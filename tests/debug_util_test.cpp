#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

TEST(cTorchDebugUtilTest, testGetRecord) {
  CTorchTensor *tensor = (CTorchTensor *)malloc(sizeof(CTorchTensor));
  MemoryRecord *record = cth_add_mem_record(tensor);
  EXPECT_EQ(tensor, record->addr);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_ALLOCATED, record->status);

  MemoryRecord *record_dup = cth_add_mem_record(tensor);
  EXPECT_EQ(record_dup, record);
}

TEST(cTorchDebugUtilTest, testAddRecord) {
  CTorchTensor *tensor = (CTorchTensor *)malloc(sizeof(CTorchTensor));
  MemoryRecord *record = cth_add_mem_record(tensor);
  MemoryRecord *record_fetch = cth_get_mem_record(tensor);
  EXPECT_EQ(record, record_fetch);

  CTorchTensor *tensor_2 = (CTorchTensor *)malloc(sizeof(CTorchTensor));
  MemoryRecord *record_fetch_2 = cth_get_mem_record(tensor_2);
  EXPECT_EQ(record_fetch_2, nullptr);
}