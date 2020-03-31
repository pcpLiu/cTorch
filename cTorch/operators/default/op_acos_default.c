#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

// clang-format off
#define _cth_acos(x)                                                           \
  _Generic((x),                                                                \
    float: acos,                                                               \
    double: acos,                                                              \
    int16_t: acos,                                                    \
    int32_t: acos,                                                    \
    int64_t: acos,                                                    \
    uint8_t: acos                                                              \
  )(x)
// clang-format on

/*
  Computes the element-wise acos value of the given input tensor.

  If calculated value is NaN, program either exit or save NaN value to output
  tensor. Global variable `CTH_NAN_EXIT` controls this.

  Does not support `CTH_TENSOR_DATA_TYPE_BOOL`.
*/
void op_acos_cpu(CTorchOperator *op) {
  FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  ListItem(CTorchTensor) *in = op->in_bound_tensors->head;
  ListItem(CTorchTensor) *out = op->out_bound_tensors->head;
  int64_t N = in->data->meta_info->n_elements;
  _cpu_1d_map_no_bool(in->data->values, out->data->values,
                      in->data->meta_info->data_type, N, acos);
}