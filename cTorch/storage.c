#include "cTorch/storage.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <string.h>

cth_impl_new_list_item_func(CTHTensor);
cth_impl_new_list_func(CTHTensor);
cth_impl_insert_list_func(CTHTensor);
cth_impl_list_contains_data_func(CTHTensor);
cth_impl_list_contains_item_func(CTHTensor);
cth_impl_list_at_func(CTHTensor);
cth_impl_list_pop_func(CTHTensor);
cth_impl_free_list_func(CTHTensor);
cth_impl_free_list_deep_func(CTHTensor);

cth_impl_new_array_func(CTHTensor);
cth_impl_array_at_func(CTHTensor);
cth_impl_array_set_func(CTHTensor);
cth_impl_free_array_deep_func(CTHTensor);

size_t cth_tensor_data_size(CTHTensor *tensor) {
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

void CTH_FORCE_TENSOR_DIMENSION(
    CTHTensor *tensor,
    cth_tensor_dim_t *target_dims,
    cth_tensor_dim_t target_n_dim) {
  // n_dim
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
    FAIL_EXIT(CTH_LOG_ERR, "CTH_FORCE_TENSOR_DIMENSION failes.");
  }
}

bool cth_tensor_name_match(CTHTensor *tensor, const char *target_name) {
  return strcmp(tensor->meta_info->tensor_name, target_name) == 0;
}

void CTH_FORCE_TENSOR_NAME(CTHTensor *tensor, const char *target_name) {
  if (!cth_tensor_name_match(tensor, target_name)) {
    // TODO: better logging
    FAIL_EXIT(CTH_LOG_ERR, "CTH_FORCE_TENSOR_NAME fails.");
  }
}

void cth_tensor_set_name(CTHTensor *tensor, const char *target_name) {
  char *name = NULL;
  // asprintf(&name, target_name);
  cth_asprintf(&name, target_name);
  tensor->meta_info->tensor_name = name;
}

void *cth_tensor_ptr_offset(CTHTensor *tensor, cth_tensor_dim_t n_elements) {
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
  } else {
    FAIL_EXIT(CTH_LOG_ERR, "Unsupported CTH_TENSOR_DATA_TYPE");
  }
}

void struct_deep_free(CTHTensorMeta)(CTHTensorMeta *meta) {
  FAIL_NULL_PTR(meta);

  FREE_SOFT(meta->dims);
  FREE_SOFT(meta->tensor_name);
  FREE(meta);
}

void struct_deep_free(CTHTensor)(CTHTensor *tensor) {
  FAIL_NULL_PTR(tensor);

  if (!tensor->meta_info->is_sharded) {
    FREE_SOFT(tensor->values);
  }

  if (tensor->meta_info != NULL) {
    struct_deep_free(CTHTensorMeta)(tensor->meta_info);
  }

  FREE(tensor);
}

void CTH_FORCE_TENSOR_NUM_ELEMENTS(
    CTHTensor *tensor, const cth_tensor_dim_t target_n) {
  FAIL_NULL_PTR(tensor);

  if (target_n != tensor->meta_info->n_elements) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "Tensor required %ld elements, but it contains %ld",
        target_n,
        tensor->meta_info->n_elements);
  }
}

void CTH_FORCE_TENSOR_TYPES(
    CTHTensor *tensor, CTH_TENSOR_DATA_TYPE *types, cth_array_index_t n_types) {
  FAIL_NULL_PTR(tensor);
  FAIL_NULL_PTR(types);

  CTH_TENSOR_DATA_TYPE data_type = tensor->meta_info->data_type;
  bool match = false;
  for (int i = 0; i < n_types; i++) {
    if (types[i] == data_type) {
      match = true;
      break;
    }
  }

  if (!match) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "CTH_FORCE_TENSOR_TYPES failed. Type %d is not supported.",
        data_type);
  }
}

cth_tensor_dim_t cth_tensor_reduce_startoffset(
    CTHTensor *tensor,
    cth_tensor_dim_t *index_dims,
    const cth_tensor_dim_t reduce_dim) {
  if (reduce_dim == -1) {
    return 0;
  }

  FAIL_NULL_PTR(tensor);
  FAIL_NULL_PTR(index_dims);

  cth_tensor_dim_t *tensor_dims = tensor->meta_info->dims;
  cth_tensor_dim_t tensor_n_dim = tensor->meta_info->n_dim;
  cth_tensor_dim_t offset = 0, i = 0;
  while (i < tensor_n_dim) {
    if (reduce_dim == i) {
      i++;
      continue;
    }

    cth_tensor_dim_t n_eles_after = 1;
    cth_tensor_dim_t j = i + 1;
    // CTH_LOG(CTH_LOG_ERR, "i,: %d, j: %d", i, j);
    while (j < tensor_n_dim) {
      n_eles_after = n_eles_after * tensor_dims[j];
      j++;
    }

    cth_tensor_dim_t index_dim_i = (i > reduce_dim ? i - 1 : i);
    offset = offset + n_eles_after * index_dims[index_dim_i];
    i++;
    // CTH_LOG(
    //     CTH_LOG_ERR,
    //     "i: %d, reduce_dim:%d, tensor_n_dim: %d, offset:%d, n_eles_after: %d,
    //     " "index_dims[index_dim_i]: %d", i, reduce_dim, tensor_n_dim, offset,
    //     n_eles_after,
    //     index_dims[index_dim_i]);
  }
  // CTH_LOG(CTH_LOG_ERR, "~~~~end cth_tensor_reduce_startoffset");
  return offset;
}

cth_tensor_dim_t cth_tensor_reduce_inneroffset(
    const CTHTensor *tensor, cth_tensor_dim_t reduce_dim) {
  FAIL_NULL_PTR(tensor);

  if (reduce_dim == -1) {
    return 1;
  }

  cth_tensor_dim_t offset = 1;
  cth_tensor_dim_t i = reduce_dim + 1;
  cth_tensor_dim_t *tensor_dims = tensor->meta_info->dims;
  while (i < tensor->meta_info->n_dim) {
    offset = offset * tensor_dims[i];
    i++;
  }

  return offset;
}

void cth_tensor_get_reduce_index(
    const CTHTensor *tensor,
    cth_tensor_dim_t group_index,
    cth_tensor_dim_t reduce_dim,
    cth_tensor_dim_t *result) {
  if (reduce_dim == -1) {
    result[0] = 0;
    return;
  }

  /**
   * Convert from group_index to reduce index list:
   *   - From biggest dimension to smallest dimension
   *   - On each dimension, do div
   */

  cth_tensor_dim_t *dims = tensor->meta_info->dims;
  cth_tensor_dim_t n_dim = tensor->meta_info->n_dim;
  for (cth_tensor_dim_t i = 0; i < n_dim; i++) {
    if (i == reduce_dim) {
      continue;
    }

    cth_tensor_dim_t n_eles_after = 1;
    cth_tensor_dim_t j = i + 1;
    while (j < n_dim) {
      if (j != reduce_dim) {
        n_eles_after = n_eles_after * dims[j];
      }
      j++;
    }

    cth_tensor_dim_t reduce_index_i = (i > reduce_dim ? i - 1 : i);
    if (reduce_index_i == n_dim - 2) {
      result[reduce_index_i] = group_index;
    } else {
      result[reduce_index_i] = group_index / n_eles_after;
      group_index = group_index % n_eles_after;
    }
  }
}
