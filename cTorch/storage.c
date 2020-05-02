#include "cTorch/storage.h"

impl_new_new_list_item_func(CTorchTensor);
impl_new_list_func(CTorchTensor);
impl_insert_list_func(CTorchTensor);
impl_list_contains_data_func(CTorchTensor);
impl_list_contains_item_func(CTorchTensor);
impl_list_at_func(CTorchTensor);

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

void FORCE_TENSOR_DIMENSION(CTorchTensor *tensor, tensor_dim *target_dims) {
  // n_dim
  tensor_dim target_n_dim = sizeof(target_dims) / sizeof(target_dims[0]);
  bool match_n_dim = (tensor->meta_info->n_dim == target_n_dim);

  // dims
  bool match_dims = true;
  if (match_n_dim) {
    match_n_dim = true;
    for (int i = 0; i < target_n_dim; i++) {
      if (tensor->meta_info->dim_size_list[i] != target_dims[i]) {
        match_dims = false;
        break;
      }
    }
  }

  if (!match_dims || !match_n_dim) {
    // TODO: better logging
    FAIL_EXIT(CTH_LOG_STR, "FORCE_TENSOR_DIMENSION failes.");
  }
}

bool tensor_name_match(CTorchTensor *tensor, const char *target_name) {
  return strcmp(tensor->meta_info->tensor_name, target_name) == 0;
}

void FORCE_TENSOR_NAME(CTorchTensor *tensor, const char *target_name) {
  if (!tensor_name_match(tensor, target_name)) {
    // TODO: better logging
    FAIL_EXIT(CTH_LOG_STR, "FORCE_TENSOR_NAME fails.");
  }
}

void tensor_set_name(CTorchTensor *tensor, const char *target_name) {
  char *name = NULL;
  asprintf(&name, target_name);
  tensor->meta_info->tensor_name = name;
}
