#ifndef C_TORCH_COMMON_H
#define C_TORCH_COMMON_H

#ifndef C_TORCH_PUBLIC_ONLY /* Begin internal APIs */

#include <stdlib.h>

#define MALLOC malloc_with_null_check

typedef struct {
  char *name;

} CTorchName;

void FAIL_NULL_PTR(void *);

void FAIL_EXIT(char *const);

void *malloc_with_null_check(size_t size);

#endif /* End internal APIs */

#endif /* COMMON_H */
