#include "cTorch/mem_util.h"
#include "cTorch/logger_util.h"

#ifdef CTH_TEST_DEBUG
#include "cTorch/debug_util.h"
#endif

#include <stdlib.h>

void *cth_malloc(size_t size) {
  void *mem = malloc(size);
  FAIL_NULL_PTR(mem);

#ifdef CTH_TEST_DEBUG
  cth_add_mem_record(mem);
#endif

  return mem;
}

void cth_free(void **ptr) {
  if (*ptr == NULL) {
    FAIL_EXIT(CTH_LOG_ERR, "Trying to free a NULL pointer.");
  }

#ifdef CTH_TEST_DEBUG
  MemoryRecord *record = cth_get_mem_record(*ptr);
  FAIL_NULL_PTR(record);
  record->status = CTH_MEM_RECORD_STATUS_FREED;
#endif

  free(*ptr);
  *ptr = NULL;
}