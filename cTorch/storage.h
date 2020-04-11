#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include "cTorch/common.h"
#include "cTorch/consts.h"
#include "cTorch/list_d.h"
#include <stdint.h>

/*
  Tensor dimension size data type
*/
#define tensor_dim uint32_t

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
  tensor_dim n_dim;

  // Dimension list
  tensor_dim *dim_size_list;

  // Number of elements
  uint64_t n_elements;

  // Alignment size of this storage
  uint16_t align_size;

  // Tensor type: normal or params
  CTH_TENSOR_TYPE type;

  // For CTH_TENSOR_TYPE_PARAM type node, this is parameter name.
  // As for other types, this is a optiona field and could ne null.
  const char *tensor_name;
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
  Set tensor'S name. This function directly overrides the tensor's name.

  Note: this function will copy `target_name`. It is safe to release
  `target_name` after calling.
*/
void tensor_set_name(CTorchTensor *tensor, const char *target_name);

/*
  Get tensor's data type size.

  Note: alignment is NOT included.
*/
size_t tensor_data_size(CTorchTensor *tensor);

/*
  Check if a tensor's name match target name.
*/
bool tensor_name_match(CTorchTensor *tensor, const char *target_name);

/*
  Check if given tensor has target dimension.
  FAIL_EXIT if not match.
*/
void FORCE_TENSOR_DIMENSION(CTorchTensor *tensor, tensor_dim *target_dims);

/*
  Check if given tensor has target name.
  FAIL_EXIT if not match.
*/
void FORCE_TENSOR_NAME(CTorchTensor *tensor, const char *target_name);

// List utils
def_list_item(CTorchTensor);
def_list(CTorchTensor);
declare_new_list_item_func(CTorchTensor);
declare_new_list_func(CTorchTensor);
declare_insert_list_func(CTorchTensor);
declare_list_contains_data_func(CTorchTensor);
declare_list_contains_item_func(CTorchTensor);

#endif /* STORAGE_H */
