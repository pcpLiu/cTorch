#ifndef C_TORCH_COMMON_H
#define C_TORCH_COMMON_H

#include <stdlib.h>
#include <string.h>

#ifndef C_TORCH_PUBLIC_ONLY /* Begin internal APIs */

/*
  Commonly used system function.
*/
#define MALLOC malloc_with_null_check
#define FREE free_with_nullify
#define MEMCPY memcpy

#define true 1
#define false 0

typedef struct {
  char *name;

} CTorchName;

void FAIL_NULL_PTR(void *);

void FAIL_EXIT(char *const);

void *malloc_with_null_check(size_t);

void *free_with_nullify(void *);

#endif /* End internal APIs */

#endif /* COMMON_H */
