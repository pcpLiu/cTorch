#ifndef CTH_OPERATOR_H
#define CTH_OPERATOR_H

#include "consts.h"
#include "storage.h"
#include <uuid/uuid.h>

typedef struct {
  uuid_t uuid;
  CTH_OPERATOR_ID op_id;
  CTorchTensor *param_tensors;
  CTorchTensor *in_bound_tensors;
  CTorchTensor *out_bound_tensors;
} CTorchOperator;

#endif /* OPERATOR_H */
