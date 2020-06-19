#ifndef CTH_OPERATOR_H
#define CTH_OPERATOR_H

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"
#include "cTorch/storage.h"

typedef struct CTorchOperator {
  CTH_OP_ID op_id;                         /* Operator ID */
  Array(CTorchTensor) * in_bound_tensors;  /* List of input tensors. It includes
                                             inputs, weight and arguments. */
  Array(CTorchTensor) * out_bound_tensors; /* List of output tensors */
  bool is_sharded;                         /* If op is a sharded one */
} CTorchOperator;

// List utils for CTorchOperator
def_list_item(CTorchOperator);
def_list(CTorchOperator);
declare_new_list_item_func(CTorchOperator);
declare_new_list_func(CTorchOperator);
declare_insert_list_func(CTorchOperator);
declare_list_at_func(CTorchOperator);
declare_list_pop_func(CTorchOperator);
declare_free_list_func(CTorchOperator);
declare_free_list_deep_func(CTorchOperator);

/*
  Check if # of in_bound_tensors == # of out_bound_tensors for given
  operator. If not, exit and give error info.
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
  Call FAIL_EXIT if set fail_exit to true and not found.
*/
CTorchTensor *
get_output_by_name(CTorchOperator *op, const char *name, bool fail_exit);

/**
 * Deep free an operator. For inbound and outbound list, it will call
 * free_list_deep(T)()
 */
void struct_deep_free(CTorchOperator)(CTorchOperator *op);

#endif /* OPERATOR_H */
