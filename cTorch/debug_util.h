#ifndef CTH_DEBUG_UTIL_H
#define CTH_DEBUG_UTIL_H

#include <stdlib.h>

#include "cTorch/list_d.h"

/**
 * Use to track memory allocation & free. Only used in debug test
 */
typedef struct MemoryRecord {
  void *addr; /* Address alocated */
  uint8_t status; /* 0 - freed; 1 - allocated */
} MemoryRecord;

def_list_item(MemoryRecord);
def_list(MemoryRecord);
declare_new_list_func(MemoryRecord);
declare_insert_list_func(MemoryRecord);
declare_list_at_func(MemoryRecord);

extern List(MemoryRecord) CTH_MEM_RECORDS;

/**
 * Get memory record based on allocated mem addres. If not found, it returns
 * NULL.
 */
MemoryRecord *get_mem_record(void *ptr);

#endif /* DEBUG_UTIL_H */
