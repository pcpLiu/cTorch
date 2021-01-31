#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * @brief 2D replication padding.
 *
 * All the variables are defined in `_cth_padding_flow_2d`
 *
 */
#define _cth_replicate_pad_2d()                                                \
  do {                                                                         \
    for (cth_tensor_dim_t i = 0; i < padding_left; i++) {                      \
      out_ptr[out_offset + i] = in_ptr[in_offset];                             \
    }                                                                          \
    for (cth_tensor_dim_t i = 0; i < padding_right; i++) {                     \
      out_ptr[out_offset + padding_left + in_x_dim + i] =                      \
          in_ptr[in_offset + in_x_dim - 1];                                    \
    }                                                                          \
                                                                               \
    if (padding_whole_row) {                                                   \
      for (cth_tensor_dim_t i = 0; i < in_x_dim; i++) {                        \
        out_ptr[out_offset + padding_left + i] = in_ptr[in_offset + i];        \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Pads the input tensor using replication of the input boundary.
 *
 * @param op CTHOperator operator
 *
 * @par Inputs & outputs:
 *   - # of input: 1
 *    - 0: input tensor, [batch (b), channel (z), height (y), length (x)]
 *   - # of output: 1
 *    - 0: output tensor, [batch (b), channel (z), height (y), length (x)]
 *
 * @par Op arguments:
 *    - CTH_PARAM_TYPE_PADDING_D4, [left, right, top, bottom]
 */
void op_ReplicationPad2d_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  _cth_padding_generic_2d(op, _cth_replicate_pad_2d);
}
