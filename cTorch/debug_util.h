#ifndef CTH_DEBUG_UTIL_H
#define CTH_DEBUG_UTIL_H

#include <stdlib.h>

#include "cTorch/list_d.h"

/**
 * Memory status
 *    - CTH_MEM_RECORD_STATUS_ALLOCATED: allocated
 *    - CTH_MEM_RECORD_STATUS_FREED: freed
 */
typedef enum CTH_MEM_RECORD_STATUS {
  CTH_MEM_RECORD_STATUS_ALLOCATED,
  CTH_MEM_RECORD_STATUS_FREED,
} CTH_MEM_RECORD_STATUS;

/**
 * Use to track memory allocation & free. Only used in debug test
 */
typedef struct MemoryRecord {
  void *addr; /* Address alocated */
  CTH_MEM_RECORD_STATUS status; /* status */
  struct MemoryRecord *next; /* Next memory record */
} MemoryRecord;

/**
 * Global variable. List of allocated memory records. It always points to the
 * 1st dummy node.
 */
extern MemoryRecord *CTH_MEM_RECORDS;

/**
 * Add an allocated addr to track.
 *
 * Params:
 *    - ptr: void*, allocated address
 *
 * Returns:
 *    The newly added memory record
 *
 * Note:
 *    If the address already in records, it will return existing memory record
 */
MemoryRecord *cth_add_mem_record(void *ptr);

/**
 * Fetch memory record with given mem address.
 *
 * Params:
 *    - ptr: void*, allocated address. If ptr is NULL, function FAIL_NULL_PTR.
 *
 * Returns:
 *    record: MemoryRecord*, memory record if found.
 *
 * Note:
 *    NULL will be returned if memory record is not found
 */
MemoryRecord *cth_get_mem_record(void *ptr);

#endif /* DEBUG_UTIL_H */