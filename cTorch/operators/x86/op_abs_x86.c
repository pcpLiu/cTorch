#include <math.h>

#include "cTorch/operators/x86/op_list.h"
#include "cTorch/operators/x86/x86_common.h"

/*
  Computes the element-wise absolute value of the given input tensor.
*/
void op_abs_x86(CTorchOperator *op) {
  FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op);

  ListItem(CTorchTensor) *in = op->in_bound_tensors->head;
  ListItem(CTorchTensor) *out = op->out_bound_tensors->head;
  int64_t N = in->data->meta_info->n_elements;

  if (in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_BOOL ||
      in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_INT_8 ||
      in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_INT_16 ||
      in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    x86_1d_map(in->data->values, out->data->values,
               in->data->meta_info->data_type, N, abs);
  } else if (in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    x86_1d_map(in->data->values, out->data->values,
               in->data->meta_info->data_type, N, llabs);
  } else if (in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
             in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    x86_1d_map(in->data->values, out->data->values,
               in->data->meta_info->data_type, N, fabs);
  } else if (in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    x86_1d_map(in->data->values, out->data->values,
               in->data->meta_info->data_type, N, fabsl);
  } else if (in->data->meta_info->data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    MEMCPY(out->data->values, in->data->values, tensor_data_size(in->data));
  }
}
