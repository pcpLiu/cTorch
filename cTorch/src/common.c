#include "common.h"
#include <stdio.h>
#include <stdlib.h>

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