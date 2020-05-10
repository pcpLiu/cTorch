#include "cTorch/debug_util.h"
#include "cTorch/list_d.h"

MemoryRecord *CTH_MEM_RECORDS = &(MemoryRecord){
    .addr = NULL, .status = CTH_MEM_RECORD_STATUS_ALLOCATED, .next = NULL};

/**
 * Number of records we are tracking (does not include 1st dummy node)
 */
static uint32_t CTH_N_RECORDS = 0;

MemoryRecord *cth_add_mem_record(void *ptr) {
  /* Check existing */
  MemoryRecord *exist = cth_get_mem_record(ptr);
  if (exist != NULL) {
    return exist;
  }

  MemoryRecord *record = malloc(sizeof(MemoryRecord));
  record->addr = ptr;
  record->status = CTH_MEM_RECORD_STATUS_ALLOCATED;
  record->next = NULL;

  /* Put new record to the end of the list */
  MemoryRecord *item = CTH_MEM_RECORDS;
  while (item->next != NULL) {
    item = item->next;
  }
  item->next = record;
  CTH_N_RECORDS++;

  return record;
}

MemoryRecord *cth_get_mem_record(void *ptr) {
  FAIL_NULL_PTR(ptr);

  MemoryRecord *record = CTH_MEM_RECORDS->next; // 1st is dummy node
  bool found = false;
  while (record != NULL) {
    if (ptr == record->addr) {
      found = true;
      break;
    } else {
      record = record->next;
    }
  }

  return found ? record : NULL;
}