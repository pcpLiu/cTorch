#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include <stdint.h>

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"
#include "cTorch/list_d.h"

/**
 * CTHTensorMeta
 * This struct contains Meta information of a tensor.
 */
typedef struct CTHTensorMeta {
  uint16_t value_size_of;         /* Element size */
  CTH_TENSOR_DATA_TYPE data_type; /* Value data type */
  cth_tensor_dim_t n_dim;         /* Number of dimensions */
  cth_tensor_dim_t *dims;         /* Dimension array */
  cth_tensor_dim_t n_elements;    /* Number of elements */
  uint16_t align_size;            /* Alignment size of this storage */
  CTH_TENSOR_TYPE type;           /* Tensor type: normal or params */
  CTH_TENSOR_DEVICE device;       /* Device where tensor lives */
  bool is_sharded;   /* If this tensor is a sharding piece of another tensor */
  char *tensor_name; /* For CTH_TENSOR_TYPE_PARAM type node, this is parameter
                        name. As for other types, this is an optiona field and
                        could be null */
} CTHTensorMeta;

/**
 * Deep free a meta info.
 *
 * Note:
 *    If pointer is NULL, error raised and exit.
 */
void struct_deep_free(CTHTensorMeta)(CTHTensorMeta *meta_info);

/**
 * Tensor struct.
 *
 * Note:
 *    - The tensor layout is using Row-major
 */
typedef struct CTHTensor {
  CTHTensorMeta *meta_info; /* Meta info */
  void *values;             /* Tensor values */
} CTHTensor;

/**
 * Deep free a tensor. Function name follows pattern defined in list_d.h
 * If the tensor is a sharded one, function will not release value block.
 *
 * Note:
 *    If pointer is NULL, error raised and exit.
 */
void struct_deep_free(CTHTensor)(CTHTensor *tensor);

// List utils for CTHTensor
cth_def_list_item(CTHTensor);
def_list(CTHTensor);
cth_declare_new_list_item_func(CTHTensor);
cth_declare_new_list_func(CTHTensor);
cth_declare_insert_list_func(CTHTensor);
cth_declare_list_contains_data_func(CTHTensor);
cth_declare_list_contains_item_func(CTHTensor);
cth_declare_list_at_func(CTHTensor);
cth_declare_list_pop_func(CTHTensor);
cth_declare_free_list_func(CTHTensor);
cth_declare_free_list_deep_func(CTHTensor);

// Array macros
cth_def_array(CTHTensor);
cth_declare_new_array_func(CTHTensor);
cth_declare_array_at_func(CTHTensor);
cth_declare_array_set_func(CTHTensor);
cth_declare_free_array_deep_func(CTHTensor);

/**
 * Get the pointer address by offsetting gieven tensor's ptr with given number
 * of elements.
 *
 */
void *cth_tensor_ptr_offset(CTHTensor *tensor, cth_tensor_dim_t n_elements);

/**
 * Set tensor'S name. This function directly overrides the tensor's name.
 *
 * Note: this function will copy `target_name`. It is safe to release
 * `target_name` after calling.
 */
void cth_tensor_set_name(CTHTensor *tensor, const char *target_name);

/**
 * Get tensor's data type size.
 *
 * Note: alignment is NOT included.
 */
size_t cth_tensor_data_size(CTHTensor *tensor);

/**
 * Check if a tensor's name match target name.
 */
bool cth_tensor_name_match(CTHTensor *tensor, const char *target_name);

/**
 * @brief Get starting offset on reduce action for reduce_dim_i
 *
 * @param tensor Target tensor
 * @param index_dims Dim index t oreduce
 * @param reduce_dim Which dim to reduce
 * @return cth_tensor_dim_t The ptr offset to act on
 */
cth_tensor_dim_t cth_tensor_reduce_startoffset(
    CTHTensor *tensor,
    cth_tensor_dim_t *index_dims,
    cth_tensor_dim_t reduce_dim);

/**
 * @brief  Get inner reduced elements offset on reduce action for reduce_dim_i
 *
 * @param tensor
 * @param reduce_dim
 * @param reduce_dim_i
 *
 * @return * cth_tensor_dim_t
 */
cth_tensor_dim_t cth_tensor_reduce_inneroffset(
    const CTHTensor *tensor, const cth_tensor_dim_t reduce_dim);

/**
 * @brief  Generate reduce index list for target group
 *
 * @param tensor Tensor
 * @param group_index Which reduce group
 * @param reduce_dim Which dim to reduce
 * @param result Result index list array
 */
void cth_tensor_get_reduce_index(
    const CTHTensor *tensor,
    cth_tensor_dim_t group_index,
    cth_tensor_dim_t reduce_dim,
    cth_tensor_dim_t *result);

/**
 * @brief Check if tensor has target dimensions, fail if not
 *
 * @param tensor tensor
 * @param target_dims target dims array
 * @param target_n_dim target number of dims
 */
void CTH_FORCE_TENSOR_DIMENSION(
    CTHTensor *tensor,
    cth_tensor_dim_t *target_dims,
    const cth_tensor_dim_t target_n_dim);

/**
 * Check if given tensor has target no. of elements. FAIL_EXIT if not match.
 */
void CTH_FORCE_TENSOR_NUM_ELEMENTS(
    CTHTensor *tensor, const cth_tensor_dim_t target_n);

/**
 * Check if given tensor has target name.
 * FAIL_EXIT if not match.
 */
void CTH_FORCE_TENSOR_NAME(CTHTensor *tensor, const char *target_name);

/**
 * Check if tensor has one of given types.
 *
 * Arguments:
 *    - tensor: tensor
 *    - types: array of types
 *    - n_types: no. of types in array
 */
void CTH_FORCE_TENSOR_TYPES(
    CTHTensor *tensor, CTH_TENSOR_DATA_TYPE *types, cth_array_index_t n_types);

#endif /* STORAGE_H */
