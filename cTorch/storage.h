#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include <stdint.h>

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"
#include "cTorch/list_d.h"

/**
 * CTorchTensorMeta
 * This struct contains Meta information of a tensor.
 */
typedef struct CTorchTensorMeta {
  uint16_t value_size_of;         /* Element size */
  CTH_TENSOR_DATA_TYPE data_type; /* Value data type */
  tensor_dim_t n_dim;             /* Number of dimensions */
  tensor_dim_t *dims;             /* Dimension array */
  tensor_size_t n_elements;       /* Number of elements */
  uint16_t align_size;            /* Alignment size of this storage */
  CTH_TENSOR_TYPE type;           /* Tensor type: normal or params */
  CTH_TENSOR_DEVICE device;       /* Device where tensor lives */
  bool is_sharded;   /* If this tensor is a sharding piece of another tensor */
  char *tensor_name; /* For CTH_TENSOR_TYPE_PARAM type node, this is parameter
                        name. As for other types, this is an optiona field and
                        could be null */
} CTorchTensorMeta;

/**
 * Deep free a meta info.
 *
 * Note:
 *    If pointer is NULL, error raised and exit.
 */
void struct_deep_free(CTorchTensorMeta)(CTorchTensorMeta *meta_info);

/**
 * Tensor struct.
 *
 * Note:
 *    - The tensor layout is using Row-major
 */
typedef struct CTorchTensor {
  CTorchTensorMeta *meta_info; /* Meta info */
  void *values;                /* Tensor values */
} CTorchTensor;

/**
 * Deep free a tensor. Function name follows pattern defined in list_d.h
 * If the tensor is a sharded one, function will not release value block.
 *
 * Note:
 *    If pointer is NULL, error raised and exit.
 */
void struct_deep_free(CTorchTensor)(CTorchTensor *tensor);

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
declare_free_list_deep_func(CTorchTensor);

// Array macros
def_array(CTorchTensor);
declare_new_array_func(CTorchTensor);
declare_array_at_func(CTorchTensor);
declare_array_set_func(CTorchTensor);
declare_free_array_deep_func(CTorchTensor);

/**
 * Get the pointer address by offsetting gieven tensor's ptr with given number
 * of elements.
 *
 */
void *cth_tensor_ptr_offset(CTorchTensor *tensor, tensor_size_t n_elements);

/**
 * Set tensor'S name. This function directly overrides the tensor's name.
 *
 * Note: this function will copy `target_name`. It is safe to release
 * `target_name` after calling.
 */
void cth_tensor_set_name(CTorchTensor *tensor, const char *target_name);

/**
 * Get tensor's data type size.
 *
 * Note: alignment is NOT included.
 */
size_t cth_tensor_data_size(CTorchTensor *tensor);

/**
 * Check if a tensor's name match target name.
 */
bool cth_tensor_name_match(CTorchTensor *tensor, const char *target_name);

/**
 * @brief Get starting offset on reduce action for reduce_dim_i
 *
 * @param tensor Target tensor
 * @param index_dims Dim index t oreduce
 * @param reduce_dim Which dim to reduce
 * @return tensor_dim_t The ptr offset to act on
 */
tensor_dim_t cth_tensor_reduce_startoffset(
    CTorchTensor *tensor, tensor_dim_t *index_dims, tensor_dim_t reduce_dim);

/**
 * @brief  Get inner reduced elements offset on reduce action for reduce_dim_i
 *
 * @param tensor
 * @param reduce_dim
 * @param reduce_dim_i
 * @return * tensor_dim_t
 */
tensor_dim_t cth_tensor_reduce_inneroffset(
    CTorchTensor *tensor, const tensor_dim_t reduce_dim);

/**
 * @brief Check if tensor has target dimensions, fail if not
 *
 * @param tensor tensor
 * @param target_dims target dims array
 * @param target_n_dim target number of dims
 */
void FORCE_TENSOR_DIMENSION(
    CTorchTensor *tensor,
    tensor_dim_t *target_dims,
    const tensor_dim_t target_n_dim);

/**
 * Check if given tensor has target no. of elements. FAIL_EXIT if not match.
 */
void FORCE_TENSOR_NUM_ELEMENTS(
    CTorchTensor *tensor, const tensor_size_t target_n);

/**
 * Check if given tensor has target name.
 * FAIL_EXIT if not match.
 */
void FORCE_TENSOR_NAME(CTorchTensor *tensor, const char *target_name);

/**
 * Check if tensor has one of given types.
 *
 * Arguments:
 *    - tensor: tensor
 *    - types: array of types
 *    - n_types: no. of types in array
 */
void FORCE_TENSOR_TYPES(
    CTorchTensor *tensor, CTH_TENSOR_DATA_TYPE *types, array_index_t n_types);

#endif /* STORAGE_H */
