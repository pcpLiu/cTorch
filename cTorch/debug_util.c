/**
 * Copyright 2021 Zhonghao Liu
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cTorch/debug_util.h"
#include "cTorch/list_d.h"

#include <stdint.h>

CTHMemoryRecord *CTH_MEM_RECORDS = &(CTHMemoryRecord){
    .addr = NULL, .status = CTH_MEM_RECORD_STATUS_ALLOCATED, .next = NULL};

/**
 * Number of records we are tracking (does not include 1st dummy node)
 */
static uint32_t CTH_N_RECORDS = 0;

CTHMemoryRecord *cth_add_mem_record(void *ptr) {
  /* Check existing */
  CTHMemoryRecord *exist = cth_get_mem_record(ptr);
  if (exist != NULL) {
    return exist;
  }

  /* use raw malloc avoiding of infinite loop */
  CTHMemoryRecord *record = malloc(sizeof(CTHMemoryRecord));
  record->addr = ptr;
  record->status = CTH_MEM_RECORD_STATUS_ALLOCATED;
  record->next = NULL;

  /* Put new record to the end of the list */
  CTHMemoryRecord *item = CTH_MEM_RECORDS;
  while (item->next != NULL) {
    item = item->next;
  }
  item->next = record;
  CTH_N_RECORDS++;

  return record;
}

CTHMemoryRecord *cth_get_mem_record(void *ptr) {
  FAIL_NULL_PTR(ptr);

  CTHMemoryRecord *record = CTH_MEM_RECORDS->next; // 1st is dummy node
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

int cth_get_num_unfree_records() {
  int count = 0;
  CTHMemoryRecord *record = CTH_MEM_RECORDS->next; // 1st is dummy node
  while (record != NULL) {
    if (record->status == CTH_MEM_RECORD_STATUS_ALLOCATED) {
      count++;
    }
    record = record->next;
  }

  return count;
}

void cth_print_unfree_records() {
  CTHMemoryRecord *record = CTH_MEM_RECORDS->next; // 1st is dummy node
  while (record != NULL) {
    if (record->status == CTH_MEM_RECORD_STATUS_ALLOCATED) {
      printf(
          "Unfreed record. Addr: %p, name: %s\n", record->addr, record->name);
    }
    record = record->next;
  }
}
