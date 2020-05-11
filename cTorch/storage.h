#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include <stdint.h>

#include "cTorch/common.h"
#include "cTorch/consts.h"
#include "cTorch/list_d.h"
#include "cTorch/pool.h"

/**
 * Type to denote tensor dimension
 */
#define tensor_dim_t uint32_t

/**
 * Type to denote tensor size
 */
#define tensor_size_t uint32_t

/**
 * CTorchTensorMeta
 * This struct cotnains Meta information of a tensor.
 */
typedef struct CTorchTensorMeta {
  uint8_t value_size_of; /* Element size */
  CTH_TENSOR_DATA_TYPE data_type; /* Value data type */
  tensor_dim_t n_dim; /* Number of dimensions */
  tensor_dim_t *dims; /* Dimension array */
  tensor_size_t n_elements; /* Number of elements */
  uint16_t align_size; /* Alignment size of this storage */
  CTH_TENSOR_TYPE type; /* Tensor type: normal or params */
  char *tensor_name; /* For CTH_TENSOR_TYPE_PARAM type node, this is
                               parameter name. As for other types, this is an
                               optiona field and could be null. */
} CTorchTensorMeta;

/**
 * Tensor struct.
 *
 * Note:
 *    - The tensor layout is using Row-major
 */
typedef struct CTorchTensor {
  CTorchTensorMeta *meta_info; /* Meta info */
  void *values; /* Tensor values */
} CTorchTensor;

// List utils for CTorchTensor
def_list_item(CTorchTensor);
def_list(CTorchTensor);
declare_new_list_item_func(CTorchTensor);
declare_new_list_func(CTorchTensor);
declare_insert_list_func(CTorchTensor);
declare_list_contains_data_func(CTorchTensor);
declare_list_contains_item_func(CTorchTensor);
declare_list_at_func(CTorchTensor);
declare_list_pop_func(CTorchTensor);
declare_free_list_func(CTorchTensor);

/**
 * Free a tensor deeply.
 */
void cth_free_tensor(CTorchTensor *tensor);

/**
 * Get the pointer address by offsetting gieven tensor's ptr with geiven number
 * of elements.
 *
 */
void *cth_tensor_ptr_offset(CTorchTensor *tensor, tensor_size_t n_elements);

/*
  Set tensor'S name. This function directly overrides the tensor's name.

  Note: this function will copy `target_name`. It is safe to release
  `target_name` after calling.
*/
void cth_tensor_set_name(CTorchTensor *tensor, const char *target_name);

/*
  Get tensor's data type size.

  Note: alignment is NOT included.
*/
size_t cth_tensor_data_size(CTorchTensor *tensor);

/*
  Check if a tensor's name match target name.
*/
bool cth_tensor_name_match(CTorchTensor *tensor, const char *target_name);

/*
  Check if given tensor has target dimension.
  FAIL_EXIT if not match.
*/
void FORCE_TENSOR_DIMENSION(CTorchTensor *tensor, tensor_dim_t *target_dims);

/*
  Check if given tensor has target name.
  FAIL_EXIT if not match.
*/
void FORCE_TENSOR_NAME(CTorchTensor *tensor, const char *target_name);

#endif /* STORAGE_H */
