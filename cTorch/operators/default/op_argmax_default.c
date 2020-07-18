#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * 1. Get pre offset and post offset
 * 2. Search
 */
#define _cth_argmanx_serach(                                                   \
    in_ptr, out_ptr, dims, n_dim, reduce_size, input_data_type, i)             \
  do {                                                                         \
                                                                               \
  } while (0)

/**
 * 1. Get reduce dim
 * 2. call _cth_argmanx_flat_serach
 */
#define _cth_argmax(op, input_data_type)                                       \
  do {                                                                         \
    CTorchParam *dim_param =                                                   \
        cth_get_param_by_type(op, CTH_PARAM_TYPE_DIM_INT32, true);             \
    tensor_dim_t dim = (tensor_dim_t)dim_param->data.dim;                      \
    tensor_size_t reduce_size = in->meta_info->dims[dim];                      \
                                                                               \
    CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);        \
    CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);      \
    input_data_type *in_ptr = (input_data_type *)in->values;                   \
    int64_t *out_ptr = (int64_t *)out->values;                                 \
                                                                               \
    for (tensor_size_t reduce_i = 0; reduce_i < reduce_size; reduce_i++) {     \
      _cth_argmanx_serach(                                                     \
          in_ptr,                                                              \
          out_ptr,                                                             \
          in->meta_info->dims,                                                 \
          in->meta_info->n_dim,                                                \
          reduce_size,                                                         \
          input_data_type,                                                     \
          i);                                                                  \
    }                                                                          \
  } while (0)

/**
 * @brief Returns the indices of the maximum values of a tensor across a
 * dimension.
 *
 * @par In this implementation, keepdim is always false
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
  _cpu_generic_compute(op, _cth_argmax, in->meta_info->data_type);
}
