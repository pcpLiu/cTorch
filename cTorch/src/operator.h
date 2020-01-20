#ifndef CTH_OPERATOR_H
#define CTH_OPERATOR_H

#include "consts.h"
#include "storage.h"
#include <uuid/uuid.h>

typedef struct {
  CTH_OPERATOR_ID op_id;
  List(CTorchTensor) * param_tensors;
  List(CTorchTensor) * in_bound_tensors;
  List(CTorchTensor) * out_bound_tensors;
} CTorchOperator;

#endif /* OPERATOR_H */
