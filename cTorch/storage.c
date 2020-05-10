#include "cTorch/storage.h"

#include <string.h>

impl_new_list_item_func(CTorchTensor);
impl_new_list_func(CTorchTensor);
impl_insert_list_func(CTorchTensor);
impl_list_contains_data_func(CTorchTensor);
impl_list_contains_item_func(CTorchTensor);
impl_list_at_func(CTorchTensor);
impl_list_pop_func(CTorchTensor);
impl_free_list_func(CTorchTensor);

size_t cth_tensor_data_size(CTorchTensor *tensor) {
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

void FORCE_TENSOR_DIMENSION(CTorchTensor *tensor, tensor_dim_t *target_dims) {
  // n_dim
  tensor_dim_t target_n_dim = sizeof(target_dims) / sizeof(target_dims[0]);
  bool match_n_dim = (tensor->meta_info->n_dim == target_n_dim);

  // dims
  bool match_dims = true;
  if (match_n_dim) {
    match_n_dim = true;
    for (int i = 0; i < target_n_dim; i++) {
      if (tensor->meta_info->dims[i] != target_dims[i]) {
        match_dims = false;
        break;
      }
    }
  }

  if (!match_dims || !match_n_dim) {
    // TODO: better logging
    FAIL_EXIT(CTH_LOG_ERR, "FORCE_TENSOR_DIMENSION failes.");
  }
}

bool cth_tensor_name_match(CTorchTensor *tensor, const char *target_name) {
  return strcmp(tensor->meta_info->tensor_name, target_name) == 0;
}

void FORCE_TENSOR_NAME(CTorchTensor *tensor, const char *target_name) {
  if (!cth_tensor_name_match(tensor, target_name)) {
    // TODO: better logging
    FAIL_EXIT(CTH_LOG_ERR, "FORCE_TENSOR_NAME fails.");
  }
}

void cth_tensor_set_name(CTorchTensor *tensor, const char *target_name) {
  char *name = NULL;
  asprintf(&name, target_name);
  tensor->meta_info->tensor_name = name;
}

void *cth_tensor_ptr_offset(CTorchTensor *tensor, tensor_size_t n_elements) {
  if (n_elements == 0)
    return tensor->values;

  CTH_TENSOR_DATA_TYPE data_type = tensor->meta_info->data_type;
  void *ptr = tensor->values;
  if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
    return (void *)((bool *)ptr + n_elements);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    return (void *)((uint8_t *)ptr + n_elements);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_8) {
    return (void *)((int8_t *)ptr + n_elements);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    return (void *)((int16_t *)ptr + n_elements);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    return (void *)((int32_t *)ptr + n_elements);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    return (void *)((int64_t *)ptr + n_elements);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {
#ifdef BACKEND_CPU_X86
    return (void *)((float *)ptr + n_elements);
#elif BACKEND_CPU_ARM
    return (void *)((__fp16 *)ptr + n_elements);
#endif
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    return (void *)((float *)ptr + n_elements);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    return (void *)((double *)ptr + n_elements);
  }
}

void cth_free_tensor(CTorchTensor *tensor) {
  FREE((void **)&tensor->meta_info->dims);
  FREE((void **)&tensor->meta_info->tensor_name);
  FREE((void **)&tensor->meta_info);
  FREE((void **)&tensor->values);
  FREE((void **)&tensor);
}