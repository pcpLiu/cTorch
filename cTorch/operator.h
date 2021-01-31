// Copyright 2021 Zhonghao Liu
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CTH_OPERATOR_H
#define CTH_OPERATOR_H

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"
#include "cTorch/params.h"
#include "cTorch/storage.h"

typedef struct CTHOperator {
  CTH_OP_ID op_id;                         /* Operator ID */
  CTHArray(CTHTensor) * in_bound_tensors;  /* List of input tensors. It
                                             includes  inputs, weights */
  CTHArray(CTHTensor) * out_bound_tensors; /* List of output tensors */
  CTHArray(CTHParam) * params;             /* Scalar parameters */
  bool is_sharded;                         /* If op is a sharded one */
} CTHOperator;

// List utils for CTHOperator
cth_def_list_item(CTHOperator);
def_list(CTHOperator);
cth_declare_new_list_item_func(CTHOperator);
cth_declare_new_list_func(CTHOperator);
cth_declare_insert_list_func(CTHOperator);
cth_declare_list_at_func(CTHOperator);
cth_declare_list_pop_func(CTHOperator);
cth_declare_free_list_func(CTHOperator);
cth_declare_free_list_deep_func(CTHOperator);

/**
 * Check if # of in_bound_tensors == # of out_bound_tensors for given operator.
 * If not, exit and give error info.
 */
void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTHOperator *op);

/**
 * Check if operator has input tensor with given name and type
 */
void FORCE_OP_INPUT_EXIST(
    CTHOperator *op, const char *target_name, CTH_TENSOR_DATA_TYPE data_type);

/**
 * Check if operatr has a parameter with given type
 */
void FORCE_OP_PARAM_EXIST(CTHOperator *op, const CTH_PARAM_TYPE type);

/**
 * Check if operator has required number of inputs & outputs
 *
 * Arguments:
 *    - op: operator
 *    - num_input: required no. of imput tensors
 *    - num_output: required no. of output tensors
 */
void FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(
    const CTHOperator *op,
    const cth_array_index_t num_input,
    const cth_array_index_t num_output);

/**
 * Check if operator has required num of params
 */
void FORCE_OP_PARAM_NUM(
    const CTHOperator *op, const cth_array_index_t num_param);

/**
 * If any input & output tensor's datatypes is unsupported, fail this op's
 * execution.
 */
void OP_FAIL_ON_DTYPE(CTHOperator *op, CTH_TENSOR_DATA_TYPE data_type);

/**
 * Get input tensor by name.
 * Call FAIL_EXIT if set fail_exit to true and not found.
 */
CTHTensor *
cth_get_input_by_name(CTHOperator *op, const char *name, bool fail_exit);

/**
 * Call FAIL_EXIT if set fail_exit to true and not found.
 */
CTHTensor *
get_output_by_name(CTHOperator *op, const char *name, bool fail_exit);

/**
 * Get param by type. Always return the first met one.
 * It returns NULL if fail_exit is set to false.
 */
CTHParam *cth_get_param_by_type(
    CTHOperator *op, const CTH_PARAM_TYPE type, bool fail_exit);

/**
 * Deep free an operator. For inbound and outbound list, it will call
 * cth_free_list_deep(T)()
 */
void struct_deep_free(CTHOperator)(CTHOperator *op);

#endif /* OPERATOR_H */
