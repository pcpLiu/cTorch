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
typedef uint32_t array_index_t;

/**
 * Array(T) --- generic array type name
 * ArrayStruct(T) --- generic array struct
 *    - size: size
 *    - _data: array of data pointers
 * def_array(T) --- typedef a generic array for T
 */
#define Array(T) array_##T
#define ArrayStruct(T)                                                         \
  struct Array(T) {                                                            \
    array_index_t size;                                                        \
    T **_data;                                                                 \
  }
#define def_array(T) typedef ArrayStruct(T) Array(T)

/**
 * Create a new array given its size. All data pointers point to NULL.
 * We allow 0 size array.
 *
 * new_array(T) --- func name
 * declare_new_array_func(T) --- declare array at function
 * impl_new_array_func(T) --- implement array at function
 */
#define new_array(T) new_array_##T
#define _declare_new_array_func(T, func_name, array_T)                         \
  array_T *func_name(array_index_t size)
#define declare_new_array_func(T)                                              \
  _declare_new_array_func(T, new_array(T), Array(T))
#define _impl_new_array_func(T, func_name, array_T)                            \
  array_T *func_name(array_index_t size) {                                     \
    array_T *array = (array_T *)MALLOC(sizeof(array_T));                       \
    array->size = size;                                                        \
    array->_data = NULL;                                                       \
    if (size > 0) {                                                            \
      array->_data = (T **)MALLOC(sizeof(T *) * size);                         \
      for (array_index_t i = 0; i < size; i++) {                               \
        *(array->_data + i) = NULL;                                            \
      }                                                                        \
    }                                                                          \
    return array;                                                              \
  }
#define impl_new_array_func(T) _impl_new_array_func(T, new_array(T), Array(T))

#define array_set(T) array_set_##T
#define _declare_array_set_func(T, func_name, array_T)                         \
  void func_name(array_T *array, array_index_t i, T *val)
#define declare_array_set_func(T)                                              \
  _declare_array_set_func(T, array_set(T), Array(T))
#define _impl_array_set_func(T, func_name, array_T)                            \
  void func_name(array_T *array, array_index_t i, T *val) {                    \
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
#define impl_array_set_func(T) _impl_array_set_func(T, array_set(T), Array(T))

/**
 * Get element at index i
 *
 * array_at(T) --- func name
 * declare_array_at_func(T) --- declare array at function
 * impl_array_at_func(T) --- implement array at function
 */
#define array_at(T) array_at_##T
#define _declare_array_at_func(T, func_name, array_T)                          \
  T *func_name(array_T *array, array_index_t i)
#define declare_array_at_func(T)                                               \
  _declare_array_at_func(T, array_at(T), Array(T))
#define _impl_array_at_func(T, func_name, array_T)                             \
  T *func_name(array_T *array, array_index_t i) {                              \
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
#define impl_array_at_func(T) _impl_array_at_func(T, array_at(T), Array(T))

/**
 * Free an array. Would NOT free contained data
 *
 *  free_array(T) --- func name
 *  declare_free_array_func(T) --- declare
 *  impl_free_array_func(T) --- implementation
 */
#define free_array(T) free_array_##T
#define _declare_free_array_func(func_name, array_T)                           \
  void func_name(array_T *array);
#define declare_free_array_func(T)                                             \
  _declare_free_array_func(free_array(T), Array(T))
#define _impl_free_array_func(func_name, array_T)                              \
  void func_name(array_T *array) {                                             \
    FAIL_NULL_PTR(array);                                                      \
                                                                               \
    FREE(array->data);                                                         \
    FREE(array);                                                               \
  }
#define impl_free_array_func(T) _impl_free_array_func(free_array(T), Array(T))

/**
 * Free an array in a deep wayy. Would also free contained data
 *
 *  free_array_deep(T) --- func name
 *  declare_free_array_deep_func(T) --- declare
 *  impl_free_array_deep_func(T) --- implementation
 */
#define free_array_deep(T) free_array_deep_##T
#define _declare_free_array_deep_func(func_name, array_T)                      \
  void func_name(array_T *array);
#define declare_free_array_deep_func(T)                                        \
  _declare_free_array_deep_func(free_array_deep(T), Array(T))
#define _impl_free_array_deep_func(func_name, array_T, T)                      \
  void func_name(array_T *array) {                                             \
    FAIL_NULL_PTR(array);                                                      \
                                                                               \
    array_index_t i = 0;                                                       \
    while (i < array->size) {                                                  \
      struct_deep_free(T)(array_at(T)(array, i));                              \
      i++;                                                                     \
    }                                                                          \
    FREE_SOFT(array->_data);                                                   \
    FREE(array);                                                               \
  }
#define impl_free_array_deep_func(T)                                           \
  _impl_free_array_deep_func(free_array_deep(T), Array(T), T)

#endif /* GENERIC_ARRAY_H */
