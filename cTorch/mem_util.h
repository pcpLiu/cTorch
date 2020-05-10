#ifndef CTH_MEM_UTIL_H
#define CTH_MEM_UTIL_H

#include <stdlib.h>

/*
  Call FAIL_EXIT() if input pointer is NULL.
*/
void *cth_malloc(size_t);

/*
  Call FAIL_EXIT() if input pointer is NULL.
*/
void cth_free(void **ptr);

/*
  malloc with limited GC
*/
#define MALLOC cth_malloc

/*
  Free and set pointer to NULL
*/
#define FREE cth_free

/*
  No idea why I need this, just feel I need it
*/
#define MEMCPY memcpy

#endif /* CTH_MEM_UTIL_H */
