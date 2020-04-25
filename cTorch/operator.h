#ifndef CTH_OPERATOR_H
#define CTH_OPERATOR_H

#include "cTorch/common.h"
#include "cTorch/consts.h"
#include "cTorch/storage.h"
#include <uuid/uuid.h>

typedef struct {
  // Operator ID
  CTH_OP_ID op_id;

  // List of input tensors. It includes inputs, weight and arguments.
  List(CTorchTensor) * in_bound_tensors;

  // List of output tensors.
  List(CTorchTensor) * out_bound_tensors;
} CTorchOperator;

/*
  Check if # of in_bound_tensors == # of out_bound_tensors for given operator.
  If not, exit and give error info.
*/
void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTorchOperator *);

/*
  Check if operator has param with given name and type
*/
void FORCE_OP_PARAM_EXIST(CTorchOperator *, const char *, CTH_TENSOR_DATA_TYPE);

/*
  If any input & output tensor's datatypes is unsupported, fail this op's
  execution.
*/
void OP_FAIL_ON_DTYPE(CTorchOperator *op, CTH_TENSOR_DATA_TYPE data_type);

/*
  Get input tensor by name.

  Call FAIL_EXIT if set fail_exit to true and not found.
*/
CTorchTensor *
get_input_by_name(CTorchOperator *op, const char *name, bool fail_exit);

/*
  Get output tensor by name.

  Call FAIL_EXIT if set fail_exit to true and not found.
*/
CTorchTensor *
get_output_by_name(CTorchOperator *op, const char *name, bool fail_exit);

#endif /* OPERATOR_H */
