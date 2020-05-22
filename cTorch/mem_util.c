#include "cTorch/mem_util.h"
#include "cTorch/logger_util.h"

#ifdef CTH_TEST_DEBUG
#include "cTorch/debug_util.h"
#endif

#include <stdarg.h>
#include <stdlib.h>

void *cth_malloc(size_t size, const char *name) {
  void *mem = malloc(size);
  FAIL_NULL_PTR(mem);

#ifdef CTH_TEST_DEBUG
  MemoryRecord *record = cth_add_mem_record(mem);
  record->name = name;
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
  MemoryRecord *record = cth_get_mem_record(ptr);
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
  MemoryRecord *record = cth_get_mem_record(ptr);
  char *msg = NULL;
  asprintf(
      &msg,
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
  MemoryRecord *record = cth_add_mem_record(*strp);
#endif

  return ret;
}