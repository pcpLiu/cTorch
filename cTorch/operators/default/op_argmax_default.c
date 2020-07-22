#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_argmax(                                                           \
    in_ptr,                                                                    \
    out_ptr,                                                                   \
    input_data_type,                                                           \
    start_offset,                                                              \
    inner_offset,                                                              \
    result_offset,                                                             \
    reduce_size)                                                               \
  do {                                                                         \
    tensor_dim_t max_i = 0;                                                    \
    input_data_type max_val = in_ptr[start_offset];                            \
    for (tensor_dim_t i = 0; i < reduce_size; i++) {                           \
      input_data_type val = in_ptr[start_offset + i * inner_offset];           \
      if (val >= max_val) {                                                    \
        max_val = val;                                                         \
        max_i = i;                                                             \
      }                                                                        \
    }                                                                          \
    out_ptr[result_offset] = max_i;                                            \
  } while (0)

/**
 * @brief Returns the indices of the maximum values of a tensor across a
 * dimension.
 *
 * @par When there's multiple max values in a dim, it returns the last position
 * it meets.
 *
 * @note In this implementation, keepdim is always false.
 *
 * @param op
 *
 * Inputs & Outputs & Params:
 *    - # of inputs: 1
 *    - # of outputs: 1
 *      - Output tensor type should be `CTH_TENSOR_DATA_TYPE_INT_64`
 *    - Argument:
 *      - dim (int): the dimension to reduce. If `-1`, the argmax of
 *        the flattened input is returned.
 */
void op_argmax_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  FORCE_OP_PARAM_NUM(op, 1);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_DIM_INT32);

  CTH_TENSOR_DATA_TYPE types[1] = {
      CTH_TENSOR_DATA_TYPE_INT_64,
  };
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  FORCE_TENSOR_TYPES(out, types, 1);

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  _cpu_reduce_arg_generic(op, in->meta_info->data_type, _cth_argmax);
}
