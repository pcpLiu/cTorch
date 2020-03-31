#include "cTorch/common.h"
#include <stdio.h>

// Set default config values
bool CTH_NAN_EXIT = false;
bool CTH_LOG_ENABLE = true;
char *CTH_LOG_STR = NULL;

void FAIL_NULL_PTR(void *ptr) {
  if (ptr == NULL) {
    fprintf(stderr, "[cTorch] NULL ptr error.");
    exit(1);
  }
}

void *malloc_with_null_check(size_t size) {
  void *mem = malloc(size);
  FAIL_NULL_PTR(mem);
  return mem;
}
