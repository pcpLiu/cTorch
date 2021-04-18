// Copyright 2021 Zhonghao Liu
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include <stdarg.h>
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
void *
cth_tensor_ptr_offset(const CTHTensor *tensor, cth_tensor_dim_t n_elements);

/**
 * Set tensor'S name. This function directly overrides the tensor's name.
 *
 * Note: this function will copy `target_name`. It is safe to release
 * `target_name` after calling.
 */
void cth_tensor_set_name(const CTHTensor *tensor, const char *target_name);

/**
 * Get tensor's data type size.
 *
 * Note: alignment is NOT included.
 */
size_t cth_tensor_data_size(const CTHTensor *tensor);

/**
 * Check if a tensor's name match target name.
 */
bool cth_tensor_name_match(const CTHTensor *tensor, const char *target_name);

/**
 * @brief Get starting offset on reduce action for `reduce_dim`.
 * A tensor with dim [2, 3, 2, 4], if reduce dim is 1 and index dims is [1, 1,
 * 3], the offset should be 31. \n
 * This func is used in reduce operation to calcualte each reduce dim's starting
 * offset.
 *
 *
 * @param tensor Target tensor
 * @param index_dims Dim index t oreduce
 * @param reduce_dim Which dim to reduce
 * @return cth_tensor_dim_t The ptr offset to act on
 */
cth_tensor_dim_t cth_tensor_reduce_startoffset(
    const CTHTensor *tensor,
    const cth_tensor_dim_t *index_dims,
    cth_tensor_dim_t reduce_dim);

/**
 * @brief Get offset after a dim.
 * For tensor dim [2, 3, 4, 5] and after_dim `1`, the value will be `4*5 = 20`.
 *
 * @param tensor CTHTensor
 * @param after_dim cth_tensor_dim_t target dim
 * @return cth_tensor_dim_t
 */
cth_tensor_dim_t cth_tensor_after_dim_offset(
    const CTHTensor *tensor, cth_tensor_dim_t after_dim);

/**
 * @brief  Get inner reduced elements offset on reduce action for
 * reduce_dim_i
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
 * @brief Generate reduce index list for target group
 *
 * @param tensor Tensor
 * @param group_index Which reduce group
 * @param reduce_dim Which dim to reduce. If value is `-1`, flatten process
 * @param result Result index list array
 */
void cth_tensor_get_reduce_index(
    const CTHTensor *tensor,
    cth_tensor_dim_t group_index,
    cth_tensor_dim_t reduce_dim,
    cth_tensor_dim_t *result);

/**
 * @brief Accesss element of a tensor based on index list. Check unit test for
 * usage
 *
 * @note If index is out of boundary, function will log error and exit.
 *
 * @param tesnor CTHTensor tensor. If it is null, function will log error and
 * exit
 * @param val_ptr void* pointer to result variable. If it is null, function will
 * log error and exit
 * @param va_list __VA_ARGS__ index list
 */
void cth_tensor_at(const CTHTensor *tensor, void *val_ptr, ...);

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
    cth_tensor_dim_t target_n_dim);

/**
 * Check if given tensor has target no. of elements. FAIL_EXIT if not match.
 */
void CTH_FORCE_TENSOR_NUM_ELEMENTS(
    const CTHTensor *tensor, cth_tensor_dim_t target_n);

/**
 * Check if given tensor has target name.
 * FAIL_EXIT if not match.
 */
void CTH_FORCE_TENSOR_NAME(const CTHTensor *tensor, const char *target_name);

/**
 * @brief Check if tensor has one of given types.
 *
 * @param tensor CTHTensor* tensor
 * @param types CTH_TENSOR_DATA_TYPE* array of types
 * @param n_types cth_array_index_t number of types in types
 */
void CTH_FORCE_TENSOR_TYPES(
    const CTHTensor *tensor,
    const CTH_TENSOR_DATA_TYPE *types,
    cth_array_index_t n_types);

#endif /* STORAGE_H */
