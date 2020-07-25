#ifndef CTH_GENERIC_ARRAY_H
#define CTH_GENERIC_ARRAY_H

/**
 * This file offers utils to build and manipulate a generic array
 */

#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <stdint.h>

/**
 * Type for array index and size
 */
typedef uint32_t cth_array_index_t;

/**
 * CTHArray(T) --- generic array type name
 * CTHArrayStruct(T) --- generic array struct
 *    - size: size
 *    - _data: array of data pointers
 * cth_def_array(T) --- typedef a generic array for T
 */
#define CTHArray(T) array_##T
#define CTHArrayStruct(T)                                                      \
  struct CTHArray(T) {                                                         \
    cth_array_index_t size;                                                    \
    T **_data;                                                                 \
  }
#define cth_def_array(T) typedef CTHArrayStruct(T) CTHArray(T)

/**
 * Create a new array given its size. All data pointers point to NULL.
 * We allow 0 size array.
 *
 * cth_new_array(T) --- func name
 * cth_declare_new_array_func(T) --- declare array at function
 * cth_impl_new_array_func(T) --- implement array at function
 */
#define cth_new_array(T) new_array_##T
#define _cth_declare_new_array_func(T, func_name, array_T)                     \
  array_T *func_name(cth_array_index_t size)
#define cth_declare_new_array_func(T)                                          \
  _cth_declare_new_array_func(T, cth_new_array(T), CTHArray(T))
#define _cth_impl_new_array_func(T, func_name, array_T)                        \
  array_T *func_name(cth_array_index_t size) {                                 \
    array_T *array = (array_T *)MALLOC(sizeof(array_T));                       \
    array->size = size;                                                        \
    array->_data = NULL;                                                       \
    if (size > 0) {                                                            \
      array->_data = (T **)MALLOC(sizeof(T *) * size);                         \
      for (cth_array_index_t i = 0; i < size; i++) {                           \
        *(array->_data + i) = NULL;                                            \
      }                                                                        \
    }                                                                          \
    return array;                                                              \
  }
#define cth_impl_new_array_func(T)                                             \
  _cth_impl_new_array_func(T, cth_new_array(T), CTHArray(T))

#define cth_array_set(T) array_set_##T
#define _cth_declare_array_set_func(T, func_name, array_T)                     \
  void func_name(array_T *array, cth_array_index_t i, T *val)
#define cth_declare_array_set_func(T)                                          \
  _cth_declare_array_set_func(T, cth_array_set(T), CTHArray(T))
#define _cth_impl_array_set_func(T, func_name, array_T)                        \
  void func_name(array_T *array, cth_array_index_t i, T *val) {                \
    FAIL_NULL_PTR(array);                                                      \
                                                                               \
    if (i >= array->size) {                                                    \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR,                                                         \
          "Array index out of boundary."                                       \
          " Given index %d while array size is %d",                            \
          i,                                                                   \
          array->size);                                                        \
    }                                                                          \
                                                                               \
    *(array->_data + i) = val;                                                 \
  }
#define cth_impl_array_set_func(T)                                             \
  _cth_impl_array_set_func(T, cth_array_set(T), CTHArray(T))

/**
 * Get element at index i
 *
 * cth_array_at(T) --- func name
 * cth_declare_array_at_func(T) --- declare array at function
 * cth_impl_array_at_func(T) --- implement array at function
 */
#define cth_array_at(T) array_at_##T
#define _cth_declare_array_at_func(T, func_name, array_T)                      \
  T *func_name(array_T *array, cth_array_index_t i)
#define cth_declare_array_at_func(T)                                           \
  _cth_declare_array_at_func(T, cth_array_at(T), CTHArray(T))
#define _cth_impl_array_at_func(T, func_name, array_T)                         \
  T *func_name(array_T *array, cth_array_index_t i) {                          \
    FAIL_NULL_PTR(array);                                                      \
                                                                               \
    if (i >= array->size) {                                                    \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR,                                                         \
          "Array index out of boundary. Given index %ud while array size is "  \
          "%ud",                                                               \
          i,                                                                   \
          array->size);                                                        \
    }                                                                          \
    return *(array->_data + i);                                                \
  }
#define cth_impl_array_at_func(T)                                              \
  _cth_impl_array_at_func(T, cth_array_at(T), CTHArray(T))

/**
 * Free an array. Would NOT free contained data
 *
 *  cth_free_array(T) --- func name
 *  cth_declare_free_array_func(T) --- declare
 *  cth_impl_free_array_func(T) --- implementation
 */
#define cth_free_array(T) free_array_##T
#define _cth_declare_free_array_func(func_name, array_T)                       \
  void func_name(array_T *array);
#define cth_declare_free_array_func(T)                                         \
  _cth_declare_free_array_func(cth_free_array(T), CTHArray(T))
#define _cth_impl_free_array_func(func_name, array_T)                          \
  void func_name(array_T *array) {                                             \
    FAIL_NULL_PTR(array);                                                      \
                                                                               \
    FREE(array->data);                                                         \
    FREE(array);                                                               \
  }
#define cth_impl_free_array_func(T)                                            \
  _cth_impl_free_array_func(cth_free_array(T), CTHArray(T))

/**
 * Free an array in a deep wayy. Would also free contained data
 *
 *  cth_free_array_deep(T) --- func name
 *  cth_declare_free_array_deep_func(T) --- declare
 *  cth_impl_free_array_deep_func(T) --- implementation
 */
#define cth_free_array_deep(T) free_array_deep_##T
#define _cth_declare_free_array_deep_func(func_name, array_T)                  \
  void func_name(array_T *array);
#define cth_declare_free_array_deep_func(T)                                    \
  _cth_declare_free_array_deep_func(cth_free_array_deep(T), CTHArray(T))
#define _cth_impl_free_array_deep_func(func_name, array_T, T)                  \
  void func_name(array_T *array) {                                             \
    FAIL_NULL_PTR(array);                                                      \
                                                                               \
    cth_array_index_t i = 0;                                                   \
    while (i < array->size) {                                                  \
      struct_deep_free(T)(cth_array_at(T)(array, i));                          \
      i++;                                                                     \
    }                                                                          \
    FREE_SOFT(array->_data);                                                   \
    FREE(array);                                                               \
  }
#define cth_impl_free_array_deep_func(T)                                       \
  _cth_impl_free_array_deep_func(cth_free_array_deep(T), CTHArray(T), T)

#endif /* GENERIC_ARRAY_H */
