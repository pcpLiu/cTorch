#include "cTorch/logger_util.h"

#include <stdio.h>
#include <stdlib.h>

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
