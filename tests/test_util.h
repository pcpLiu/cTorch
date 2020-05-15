#ifndef CTH_TEST_UTL_H
#define CTH_TEST_UTL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "cTorch/c_torch.h"

CTorchTensor *create_dummy_tensor(tensor_dim_t *dims, tensor_dim_t n_dim,
                                  CTH_TENSOR_DATA_TYPE data_type, float min,
                                  float max);

/*
  Create a dummy op node with one input and one output.
  Input & output has same dimensions.
*/
CTorchNode *create_dummy_op_node(CTH_OP_ID op_id, uint32_t *dims,
                                 tensor_dim_t n_dim,
                                 CTH_TENSOR_DATA_TYPE data_type, float min,
                                 float max);

/*
  Create a dummy operator.
*/
CTorchOperator *create_dummy_op();

/*
  Create a dummy node without any information.
*/
CTorchNode *create_dummy_node();

/*
  If all values are NAN, return true.
*/
bool tensor_all_nan(CTorchTensor *);

/*
  Check element-wise equal between verify_func(input) and output
*/
#define _ele_wise_equal(type, eq_func, verify_func)                            \
  {                                                                            \
    type *input =                                                              \
        (type *)op_node->conent.op->in_bound_tensors->head->data->values;      \
    type *output =                                                             \
        (type *)op_node->conent.op->out_bound_tensors->head->data->values;     \
    uint64_t n_ele = op_node->conent.op->in_bound_tensors->head->data          \
                         ->meta_info->n_elements;                              \
    for (int i = 0; i < n_ele; i++) {                                          \
      eq_func((type)verify_func(input[i]), output[i]);                         \
    }                                                                          \
  }

#ifdef __cplusplus
}
#endif

#endif /* TEST_UTL_H */
