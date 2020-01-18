#ifndef C_TORCH_COMMON_H
#define C_TORCH_COMMON_H

#define MALLOC malloc

typedef struct {
  char *name;

} CTorchName;

void FAIL_NULL_PTR(void *);

#endif /* COMMON_H */
