#ifndef CTH_TEST_UTL_H
#define CTH_TEST_UTL_H

// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdbool.h>
#include <stdlib.h>
#include <tgmath.h>

#include "cTorch/c_torch.h"

#define CPU_CORES 4

CTHTensor *create_dummy_tensor(cth_tensor_dim_t *dims, cth_tensor_dim_t n_dim,
                               CTH_TENSOR_DATA_TYPE data_type, float min,
                               float max);

/**
 * Create a dummy op node with one input and one output.
 * Input & output has same dimensions.
 */
CTHNode *create_dummy_op_node_unary(CTH_OP_ID op_id, uint32_t *dims,
                                    cth_tensor_dim_t n_dim,
                                    CTH_TENSOR_DATA_TYPE data_type, float min,
                                    float max);

/*
  Create a dummy operator.
*/
CTHOperator *create_dummy_op(CTH_OP_ID op_id, cth_array_index_t num_inputs,
                             cth_array_index_t num_outputs);

/*
  Create a dummy operator.
*/
CTHOperator *create_dummy_op_with_param(CTH_OP_ID op_id,
                                        cth_array_index_t num_inputs,
                                        cth_array_index_t num_outputs,
                                        cth_array_index_t num_param);
/*
  Create a dummy node without any information.
*/
CTHNode *create_dummy_node(node_id_t id, cth_array_index_t inbound_size,
                           cth_array_index_t outbound_size);

/**
 * Create a dummy graph
 */
CTHGraph *create_dummy_graph(cth_array_index_t num_nodes);

/**
 * Rand flost in range
 */
float _rand_float(float min, float max);

/**
 * @brief with precision control
 *
 */
#define EXPECT_EQ_PRECISION(val_1, val_2, precision)                           \
  do {                                                                         \
    EXPECT_TRUE(abs(val_1 - val_2) <= precision);                              \
  } while (0)

/**
 * Check element-wise equal between verify_func(input) and output
 */
#define _ele_wise_equal_unary(op, type, eq_func, verify_func)                  \
  {                                                                            \
    CTHTensor *tensor_input =                                                  \
        cth_array_at(CTHTensor)(op->in_bound_tensors, 0);                      \
    type *input = (type *)tensor_input->values;                                \
    CTHTensor *tensor_output =                                                 \
        cth_array_at(CTHTensor)(op->out_bound_tensors, 0);                     \
    type *output = (type *)tensor_output->values;                              \
    cth_tensor_dim_t n_ele = tensor_input->meta_info->n_elements;              \
    for (cth_tensor_dim_t i = 0; i < n_ele; i++) {                             \
      type expect_result = (type)verify_func(input[i]);                        \
      eq_func(expect_result, output[i]);                                       \
    }                                                                          \
  }

/**
 * @brief Check element-wise equal between verify_func(input) and output with
 * libtorch
 *
 */
#define _ele_wise_equal_unary_pytorch(op, type, eq_func, eq_precision,         \
                                      torch_call)                              \
  {                                                                            \
    CTHTensor *tensor_input =                                                  \
        cth_array_at(CTHTensor)(op->in_bound_tensors, 0);                      \
    CTHTensor *tensor_output =                                                 \
        cth_array_at(CTHTensor)(op->out_bound_tensors, 0);                     \
    type *output = (type *)tensor_output->values;                              \
    cth_tensor_dim_t n_ele = tensor_input->meta_info->n_elements;              \
                                                                               \
    auto pytorch_in_tensor = create_torch_tensor(tensor_input);                \
    auto pytorch_out_tensor = torch_call(pytorch_in_tensor);                   \
    auto pytorch_result_tensor_flat = pytorch_out_tensor.reshape({n_ele});     \
    for (cth_tensor_dim_t i = 0; i < n_ele; i++) {                             \
      eq_func(pytorch_result_tensor_flat[i].item<type>(), output[i],           \
              eq_precision);                                                   \
    }                                                                          \
  }

/**
 * Check element-wise equal between verify_func(input) and output
 */
#define _ele_wise_equal_binary(op, type, eq_func, verify_func)                 \
  {                                                                            \
    CTHTensor *tensor_input_a =                                                \
        cth_array_at(CTHTensor)(op->in_bound_tensors, 0);                      \
    CTHTensor *tensor_input_b =                                                \
        cth_array_at(CTHTensor)(op->in_bound_tensors, 1);                      \
    type *input_a = (type *)tensor_input_a->values;                            \
    type *input_b = (type *)tensor_input_b->values;                            \
    CTHTensor *tensor_output =                                                 \
        cth_array_at(CTHTensor)(op->out_bound_tensors, 0);                     \
    type *output = (type *)tensor_output->values;                              \
    uint64_t n_ele = tensor_input_a->meta_info->n_elements;                    \
    for (cth_tensor_dim_t i = 0; i < n_ele; i++) {                             \
      type expect_result = verify_func(input_a[i], input_b[i]);                \
      eq_func(expect_result, output[i]);                                       \
    }                                                                          \
  }

/**
 * @brief Reduce op verify
 */
#define _reduce_op(op, in_type, out_type, pytorch_call, eq_func)               \
  {                                                                            \
    CTHTensor *tensor_input =                                                  \
        cth_array_at(CTHTensor)(op->in_bound_tensors, 0);                      \
    CTHTensor *tensor_output =                                                 \
        cth_array_at(CTHTensor)(op->out_bound_tensors, 0);                     \
    out_type *output = (out_type *)tensor_output->values;                      \
    CTHParam *dim_param =                                                      \
        cth_get_param_by_type(op, CTH_PARAM_TYPE_DIM_INT32, true);             \
    cth_tensor_dim_t reduce_dim = (cth_tensor_dim_t)dim_param->data.dim;       \
                                                                               \
    auto pytorch_in_tensor = create_torch_tensor(tensor_input);                \
    auto pytorch_result_tensor = pytorch_call(pytorch_in_tensor, reduce_dim);  \
                                                                               \
    cth_tensor_dim_t out_n_ele = tensor_output->meta_info->n_elements;         \
    auto pytorch_result_tensor_flat =                                          \
        pytorch_result_tensor.reshape({out_n_ele});                            \
    auto expect_result = (int64_t *)pytorch_result_tensor_flat.data_ptr();     \
    for (cth_tensor_dim_t i = 0; i < out_n_ele; i++) {                         \
      eq_func(expect_result[i], output[i]);                                    \
    }                                                                          \
  }

/**
 * @brief Print out sample value
 *
 * @param type
 * @param in_ptr
 * @param out_ptr
 * @param i
 */
void sample_print(CTH_TENSOR_DATA_TYPE type, void *in_ptr, void *out_ptr,
                  cth_tensor_dim_t i);

void sample_print_triple(CTH_TENSOR_DATA_TYPE type, void *in_ptr_1,
                         void *in_ptr_2, void *out_ptr, cth_tensor_dim_t i);

inline int *heap_int(int x) {
  int *ptr = (int *)MALLOC(sizeof(int));
  *ptr = x;
  return ptr;
}

// for testFreeListDeep
inline void cth_free_deep_int(int *x) { FREE(x); }

/**
 * @brief Generate random dim
 *
 * @param dims
 * @param n_dim
 */
void _rand_dims(cth_tensor_dim_t *dims, cth_tensor_dim_t n_dim,
                cth_tensor_dim_t min, cth_tensor_dim_t max);

/**
 * @brief Get reduc dims
 *
 * @param dims
 * @param n_dim
 * @param reduce_dim
 * @param reduce_dims
 */
void _get_reduce_dims(cth_tensor_dim_t *dims, cth_tensor_dim_t n_dim,
                      cth_tensor_dim_t reduce_dim,
                      cth_tensor_dim_t *reduce_dims);

/**
 * @brief Generate randin int [min, max]
 *
 * @param min
 * @param max
 * @return int
 */
int _rand_int(int min, int max);

void _print_index(cth_tensor_dim_t *dims, cth_tensor_dim_t n_dim);

// #ifdef __cplusplus
// }
// #endif

#endif /* TEST_UTL_H */
