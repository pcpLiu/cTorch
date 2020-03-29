#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include "cTorch/common.h"
#include "cTorch/consts.h"
#include "cTorch/list_d.h"
#include <stdint.h>

/*
  CTorchTensorMeta

  This struct cotnains Meta information of a tensor.
*/
typedef struct CTorchTensorMeta {
  // Element size
  uint8_t value_size_of;

  // Value data type
  CTH_TENSOR_DATA_TYPE data_type;

  // Number of dimensions
  uint16_t n_dim;

  // Dimension list
  uint32_t *dim_size_list;

  // Number of elements
  uint64_t n_elements;

  // Alignment size of this storage
  uint16_t align_size;

  // Tensor type: normal or params
  CTH_TENSOR_TYPE type;

  // For CTH_TENSOR_TYPE_PARAM type node, this is parameter name.
  // As for other types, this is a optiona field and could ne null.
  CTorchName *tensor_name;
} CTorchTensorMeta;

/*
  CTorchTensor

  This struct represents a tensor obj in a computation gprah.
*/
typedef struct CTorchTensor {
  CTorchTensorMeta *meta_info;
  void *values;
} CTorchTensor;

/*
  Get tensor data size
*/
size_t tensor_data_size(CTorchTensor *);

// List utils
def_list_item(CTorchTensor);
def_list(CTorchTensor);
declare_new_list_item_func(CTorchTensor);
declare_new_list_func(CTorchTensor);
declare_insert_list_func(CTorchTensor);
declare_list_contains_data_func(CTorchTensor);
declare_list_contains_item_func(CTorchTensor);

#endif /* STORAGE_H */