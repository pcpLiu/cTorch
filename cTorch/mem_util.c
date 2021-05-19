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

#include "cTorch/mem_util.h"
#include "cTorch/logger_util.h"

#ifdef CTH_TEST_DEBUG
#include "cTorch/debug_util.h"
#endif

#include <stdarg.h>
#include <stdlib.h>

void *cth_malloc(
    size_t size,
    const char *record_name,
    const char *file_name,
    int line_num,
    const char *func_name) {
  void *mem = malloc(size);
  FAIL_NULL_PTR(mem);

#ifdef CTH_TEST_DEBUG
  CTHMemoryRecord *record = cth_add_mem_record(mem);
  record->name = record_name;
#endif

  return mem;
}

void cth_free(
    void *ptr, const char *file_name, int line_num, const char *func_name) {
  if (ptr == NULL) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "Trying to free a NULL pointer. line: %d, function: %s, file: %s.",
        line_num,
        func_name,
        file_name);
  }

#ifdef CTH_TEST_DEBUG
  CTHMemoryRecord *record = cth_get_mem_record(ptr);
  FAIL_NULL_PTR(record);
  record->status = CTH_MEM_RECORD_STATUS_FREED;
#endif
  free(ptr);
}

void cth_free_soft(
    void *ptr, const char *file_name, int line_num, const char *func_name) {
  if (ptr == NULL) {
    return;
  }

#ifdef CTH_TEST_DEBUG
  CTHMemoryRecord *record = cth_get_mem_record(ptr);
  char *msg = NULL;
  asprintf(
      &msg,
      "cth_get_mem_record() failed. "
      "Calling from line: %d, function: %s, file: %s",
      line_num,
      func_name,
      file_name);
  FAIL_NULL_PTR_MSG(record, msg);
  record->status = CTH_MEM_RECORD_STATUS_FREED;
#endif

  free(ptr);
}

int cth_asprintf(char **strp, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vasprintf(strp, fmt, args);

#ifdef CTH_TEST_DEBUG
  CTHMemoryRecord *record = cth_add_mem_record(*strp);
#endif

  return ret;
}
