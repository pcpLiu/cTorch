#ifndef CTH_OPERATOR_H
#define CTH_OPERATOR_H

#include "cTorch/common.h"
#include "cTorch/consts.h"
#include "cTorch/storage.h"
#include <uuid/uuid.h>

typedef struct {
  CTH_OP_ID op_id;
  List(CTorchTensor) * param_tensors;
  List(CTorchTensor) * in_bound_tensors;
  List(CTorchTensor) * out_bound_tensors;
} CTorchOperator;

/*
  Check if # of in_bound_tensors == # of out_bound_tensors for given operator.
  If not, exit and give error info.
*/
void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTorchOperator *);

/*
  If any input & output tensor's datatypes is unsupported, fail this op's
  execution.
*/
void OP_FAIL_ON_DTYPE(CTorchOperator *, CTH_TENSOR_DATA_TYPE);

#endif /* OPERATOR_H */
