#ifndef CTH_TEST_UTL_H
#define CTH_TEST_UTL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "cTorch/c_torch.h"

#define CPU_CORES 4

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
CTorchOperator *create_dummy_op(array_index_t num_inputs,
                                array_index_t num_outputs);

/*
  Create a dummy node without any information.
*/
CTorchNode *create_dummy_node(node_id_t id, array_index_t inbound_size,
                              array_index_t outbound_size);

/**
 * Create a dummy graph
 */
CTorchGraph *create_dummy_graph(array_index_t num_nodes);

/*
  If all values are NAN, return true.
*/
bool tensor_all_nan(CTorchTensor *);

/*
  Check element-wise equal between verify_func(input) and output
*/
#define _ele_wise_equal(type, eq_func, verify_func)                            \
  {                                                                            \
    CTorchTensor *tensor_input =                                               \
        array_at(CTorchTensor)(op_node->conent.op->in_bound_tensors, 0);       \
    type *input = (type *)tensor_input->values;                                \
    CTorchTensor *tensor_output =                                              \
        array_at(CTorchTensor)(op_node->conent.op->out_bound_tensors, 0);      \
    type *output = (type *)tensor_output->values;                              \
    uint64_t n_ele = tensor_input->meta_info->n_elements;                      \
    for (int i = 0; i < n_ele; i++) {                                          \
      eq_func((type)verify_func(input[i]), output[i]);                         \
    }                                                                          \
  }

#ifdef __cplusplus
}
#endif

#endif /* TEST_UTL_H */
