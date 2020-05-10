#ifndef CTH_MEM_UTIL_H
#define CTH_MEM_UTIL_H

#include <stdlib.h>

/*
  Call FAIL_EXIT() if input pointer is NULL.
*/
void *malloc_with_null_check(size_t);

/*
  Call FAIL_EXIT() if input pointer is NULL.
*/
void free_with_nullify(void **ptr);

/*
  malloc with limited GC
*/
#define MALLOC malloc_with_null_check

/*
  Free and set pointer to NULL
*/
#define FREE free_with_nullify

/*
  No idea why I need this, just feel I need it
*/
#define MEMCPY memcpy

#endif /* CTH_MEM_UTIL_H */
