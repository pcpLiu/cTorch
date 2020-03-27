#include <tgmath.h>

#include "cTorch/operators/x86/op_list.h"
#include "cTorch/operators/x86/x86_common.h"

// clang-format off
#define _cth_abs(x)                                                           \
  _Generic((x),                                                                \
    float: fabs,                                                               \
    double: fabs,                                                              \
    int16_t: abs,                                                    \
    int32_t: abs,                                                    \
    int64_t: abs,                                                    \
    uint8_t: abs,                                                              \
    bool: abs                                                       \
  )(x)
// clang-format on

/*
  Computes the element-wise absolute value of the given input tensor.
*/
void op_abs_x86(CTorchOperator *op) {
  FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op);

  ListItem(CTorchTensor) *in = op->in_bound_tensors->head;
  ListItem(CTorchTensor) *out = op->out_bound_tensors->head;
  int64_t N = in->data->meta_info->n_elements;
  _x86_1d_map(in->data->values, out->data->values,
              in->data->meta_info->data_type, N, _cth_abs);
}
