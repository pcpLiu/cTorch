#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

// clang-format off
#define _cth_abs(x)                                                            \
  _Generic((x),                                                                \
    float: fabs,                                                               \
    double: fabs,                                                              \
    int16_t: abs,                                                              \
    int32_t: abs,                                                    \
    int64_t: abs,                                                    \
    uint8_t: abs,                                                              \
    bool: abs                                                       \
  )(x)
// clang-format on

/**
 * Compute the element-wise absolute value of the given input tensor.
 *
 * # of input: 1
 * # of output: 1
 */
void op_abs_cpu(CTorchOperator *op) {
  FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op);

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int64_t N = in->meta_info->n_elements;
  _cpu_1d_map(in->values, out->values, in->meta_info->data_type, N, _cth_abs);
}
