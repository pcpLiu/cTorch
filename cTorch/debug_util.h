// Copyright 2021 Zhonghao Liu
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CTH_DEBUG_UTIL_H
#define CTH_DEBUG_UTIL_H

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
typedef struct CTHMemoryRecord {
  void *addr;                   /* Address alocated */
  CTH_MEM_RECORD_STATUS status; /* status */
  struct CTHMemoryRecord *next; /* Next memory record */
  const char *name;             /* A readable name. */
} CTHMemoryRecord;

/**
 * Global variable. List of allocated memory records. It always points to the
 * 1st dummy node.
 */
extern CTHMemoryRecord *CTH_MEM_RECORDS;

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
CTHMemoryRecord *cth_add_mem_record(void *ptr);

/**
 * Fetch memory record with given mem address.
 *
 * Params:
 *    - ptr: void*, allocated address. If ptr is NULL, function FAIL_NULL_PTR.
 *
 * Returns:
 *    record: CTHMemoryRecord*, memory record if found.
 *
 * Note:
 *    NULL will be returned if memory record is not found
 */
CTHMemoryRecord *cth_get_mem_record(void *ptr);

/**
 * Get number of memory records that not freed
 *
 */
int cth_get_num_unfree_records();

/**
 * Print unfreed records
 */
void cth_print_unfree_records();

#endif /* DEBUG_UTIL_H */
