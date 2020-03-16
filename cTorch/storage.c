#include "cTorch/storage.h"

impl_new_new_list_item_func(CTorchTensor);
impl_new_list_func(CTorchTensor);
impl_insert_list_func(CTorchTensor);
impl_list_contains_data_func(CTorchTensor);
impl_list_contains_item_func(CTorchTensor);

size_t tensor_data_size(CTorchTensor *tensor) {
  size_t ele_size = 0;
  CTH_TENSOR_DATA_TYPE data_type = tensor->meta_info->data_type;
  if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    ele_size = sizeof(char);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    ele_size = sizeof(uint8_t);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_8) {
    ele_size = sizeof(int8_t);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    ele_size = sizeof(int16_t);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    ele_size = sizeof(int32_t);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    ele_size = sizeof(int64_t);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {
#ifdef BACKEND_CPU_X86
    ele_size = sizeof(float);
#elif BACKEND_CPU_ARM
    ele_size = sizeof(__fp16);
#endif
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    ele_size = sizeof(float);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    ele_size = sizeof(double);
  }

  return ele_size * tensor->meta_info->n_elements;
}