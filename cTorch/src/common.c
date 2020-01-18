#include "common.h"
#include <stdio.h>
#include <stdlib.h>

void FAIL_NULL_PTR(void *ptr) {
  if (ptr == NULL) {
    fprintf(stderr, "[cTorch] NULL ptr error.");
    exit(1);
  }
}