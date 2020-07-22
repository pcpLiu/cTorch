#ifndef CTH_OPERATOR_H
#define CTH_OPERATOR_H

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"
#include "cTorch/params.h"
#include "cTorch/storage.h"

typedef struct CTorchOperator {
  CTH_OP_ID op_id;                            /* Operator ID */
  CTHArray(CTorchTensor) * in_bound_tensors;  /* List of input tensors. It
                                             includes  inputs, weights */
  CTHArray(CTorchTensor) * out_bound_tensors; /* List of output tensors */
  CTHArray(CTorchParam) * params;             /* Scalar parameters */
  bool is_sharded;                            /* If op is a sharded one */
} CTorchOperator;

// List utils for CTorchOperator
cth_def_list_item(CTorchOperator);
def_list(CTorchOperator);
cth_declare_new_list_item_func(CTorchOperator);
cth_declare_new_list_func(CTorchOperator);
cth_declare_insert_list_func(CTorchOperator);
cth_declare_list_at_func(CTorchOperator);
cth_declare_list_pop_func(CTorchOperator);
cth_declare_free_list_func(CTorchOperator);
cth_declare_free_list_deep_func(CTorchOperator);

/**
 * Check if # of in_bound_tensors == # of out_bound_tensors for given operator.
 * If not, exit and give error info.
 */
void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTorchOperator *op);

/**
 * Check if operator has input tensor with given name and type
 */
void FORCE_OP_INPUT_EXIST(
    CTorchOperator *op,
    const char *target_name,
    CTH_TENSOR_DATA_TYPE data_type);

/**
 * Check if operatr has a parameter with given type
 */
void FORCE_OP_PARAM_EXIST(CTorchOperator *op, const CTH_PARAM_TYPE type);

/**
 * Check if operator has required number of inputs & outputs
 *
 * Arguments:
 *    - op: operator
 *    - num_input: required no. of imput tensors
 *    - num_output: required no. of output tensors
 */
void FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(
    const CTorchOperator *op,
    const array_index_t num_input,
    const array_index_t num_output);

/**
 * Check if operator has required num of params
 */
void FORCE_OP_PARAM_NUM(
    const CTorchOperator *op, const array_index_t num_param);

/**
 * If any input & output tensor's datatypes is unsupported, fail this op's
 * execution.
 */
void OP_FAIL_ON_DTYPE(CTorchOperator *op, CTH_TENSOR_DATA_TYPE data_type);

/**
 * Get input tensor by name.
 * Call FAIL_EXIT if set fail_exit to true and not found.
 */
CTorchTensor *
cth_get_input_by_name(CTorchOperator *op, const char *name, bool fail_exit);

/**
 * Call FAIL_EXIT if set fail_exit to true and not found.
 */
CTorchTensor *
get_output_by_name(CTorchOperator *op, const char *name, bool fail_exit);

/**
 * Get param by type. Always return the first met one.
 */
CTorchParam *cth_get_param_by_type(
    CTorchOperator *op, const CTH_PARAM_TYPE type, bool fail_exit);

/**
 * Deep free an operator. For inbound and outbound list, it will call
 * cth_free_list_deep(T)()
 */
void struct_deep_free(CTorchOperator)(CTorchOperator *op);

#endif /* OPERATOR_H */
