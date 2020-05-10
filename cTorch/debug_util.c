#include "cTorch/debug_util.h"
#include "cTorch/list_d.h"

impl_new_list_item_func(MemoryRecord);
impl_new_list_func(MemoryRecord);
impl_insert_list_func(MemoryRecord);
impl_list_at_func(MemoryRecord);

List(MemoryRecord) CTH_MEM_RECORDS = {
    .size = 0,
    .head = NULL,
    .tail = NULL,
};

MemoryRecord *get_mem_record(void *ptr) {
  ListItem(MemoryRecord) *item = CTH_MEM_RECORDS.head;
  bool found = false;
  MemoryRecord *record;
  for (list_index_t i = 0; i < CTH_MEM_RECORDS.size; i++) {
    record = item->data;
    if (record->addr == ptr) {
      if (record->status == 0) {
        FAIL_EXIT(
            CTH_LOG_STR,
            "Trying to free address %p, but it's already been freed.",
            ptr);
      }
      found = true;
      record->status = 0;
      break;
    }
    item = item->next_item;
  }
  return record;
}