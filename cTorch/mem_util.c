#include "cTorch/mem_util.h"
#include "cTorch/logger_util.h"

#ifdef CTH_TEST_DEBUG
#include "cTorch/debug_util.h"
#endif

#include <stdlib.h>

void *malloc_with_null_check(size_t size) {
  void *mem = malloc(size);
  FAIL_NULL_PTR(mem);

#ifdef CTH_TEST_DEBUG
  /**
   * In debug mode, record memory allocation
   */

  MemoryRecord *record = malloc(sizeof(MemoryRecord));
  record->addr = mem;
  record->status = 1;
  insert_list(MemoryRecord)(&CTH_MEM_RECORDS, record);
#endif /* CTH_TEST_DEBUG */

  return mem;
}

void free_with_nullify(void **ptr) {
  if (*ptr != NULL) {

#ifdef CTH_TEST_DEBUG
    /**
     * In debug mode, record memory deallocation
     */

    ListItem(MemoryRecord) *item = CTH_MEM_RECORDS.head;
    bool found = false;
    for (list_index_t i = 0; i < CTH_MEM_RECORDS.size; i++) {
      MemoryRecord *record = item->data;
      if (record->addr == *ptr) {
        if (record->status == 0) {
          FAIL_EXIT(
              CTH_LOG_STR,
              "Trying to free address %p, but it's already been freed.",
              *ptr);
        }
        found = true;
        record->status = 0;
        break;
      }
      item = item->next_item;
    }

    if (!found) {
      FAIL_EXIT(
          CTH_LOG_STR,
          "Trying to free address %p, but cannot find it in mem_record.",
          *ptr);
    }
#endif /* CTH_TEST_DEBUG */

    free(*ptr);
    *ptr = NULL;
  }
}