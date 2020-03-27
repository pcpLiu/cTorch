#include "cTorch/common.h"
#include <stdio.h>

bool CTH_NAN_EXIT = false;

void FAIL_NULL_PTR(void *ptr) {
  if (ptr == NULL) {
    fprintf(stderr, "[cTorch] NULL ptr error.");
    exit(1);
  }
}

void FAIL_EXIT(char *const err_msg) {
  fprintf(stderr, "[cTorch] %s", err_msg);
  exit(1);
}

void *malloc_with_null_check(size_t size) {
  void *mem = malloc(size);
  FAIL_NULL_PTR(mem);
  return mem;
}
