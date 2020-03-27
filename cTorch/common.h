#ifndef C_TORCH_COMMON_H
#define C_TORCH_COMMON_H

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifndef C_TORCH_PUBLIC_ONLY /* Begin internal APIs */

/*
  Commonly used system function.
*/
#define MALLOC malloc_with_null_check
#define FREE free_with_nullify
#define MEMCPY memcpy

typedef struct {
  char *name;

} CTorchName;

void FAIL_NULL_PTR(void *);

void FAIL_EXIT(char *const);

void *malloc_with_null_check(size_t);

void *free_with_nullify(void *);

extern bool CTH_NAN_EXIT;

/*
  Exit if val is NAN
*/
#define NAN_EXIT(val)                                                          \
  do {                                                                         \
    if (val != val && CTH_NAN_EXIT == true) {                                  \
      FAIL_EXIT("Value is NaN");                                               \
    }                                                                          \
  } while (0)

#endif /* End internal APIs */

#endif /* COMMON_H */
