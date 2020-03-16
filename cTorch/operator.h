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
  If not, exit and give error info
*/
#define FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op_ptr)                                  \
  {                                                                            \
    if (op_ptr->in_bound_tensors->size != op_ptr->out_bound_tensors->size) {   \
      FAIL_EXIT(                                                               \
          "Operator should have same numbers of input and output tensors.");   \
    }                                                                          \
  }

#endif /* OPERATOR_H */
