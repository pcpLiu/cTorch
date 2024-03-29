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

#ifndef CTH_DEFAULT_UTIL_H
#define CTH_DEFAULT_UTIL_H

#include <string.h>

/**
 * @brief Conduct unary operation element wise
 *
 * @param input_ptr: C pointer, input values
 * @param output_ptr: C pointer, output values
 * @param N: cth_tensor_dim_t, num of elements
 * @param kernel: compute function
 * @param type_t: C type
 */
#define _cpu_1d_map_elewise_unary_forloop(                                     \
    input_ptr, output_ptr, N, kernel, type_t)                                  \
  {                                                                            \
    type_t *input_t = (type_t *)input_ptr;                                     \
    type_t *output_t = (type_t *)output_ptr;                                   \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = (type_t)kernel(input_t[i]);                                \
    }                                                                          \
  }

/**
 * @brief Conduct binary operation element wise
 *
 * @param input_ptr_a: C pointer, input a values
 * @param input_ptr_b: C pointer, input b values
 * @param output_ptr: C pointer, output values
 * @param N: cth_tensor_dim_t, num of elements
 * @param kernel: compute function
 * @param type_t: C type
 */
#define _cpu_1d_map_elewise_binary_forloop(                                    \
    input_ptr_a, input_ptr_b, output_ptr, N, kernel, type_t)                   \
  {                                                                            \
    type_t *input_a_t = (type_t *)input_ptr_a;                                 \
    type_t *input_b_t = (type_t *)input_ptr_b;                                 \
    type_t *output_t = (type_t *)output_ptr;                                   \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = (type_t)kernel(input_a_t[i], input_b_t[i]);                \
    }                                                                          \
  }

/**
 * @brief Apply 1D unary kernel on input and output pointers in element wise way
 *
 * @param input_ptr: C pointer, input values
 * @param output_ptr: C pointer, output values
 * @param data_type: CTH_TENSOR_DATA_TYPE, input type
 * @param N: cth_tensor_dim_t, num of elements
 * @param kernel: compute function
 */
#define _cpu_1d_map_elewise_unary(input_ptr, output_ptr, data_type, N, kernel) \
  {                                                                            \
    if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                              \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, bool);                             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, int16_t);                          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, int32_t);                          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, int64_t);                          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, uint8_t);                          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {                   \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, float);                            \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                   \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, float);                            \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      _cpu_1d_map_elewise_unary_forloop(                                       \
          input_ptr, output_ptr, N, kernel, double);                           \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR, "Unsupported data type in _cpu_1d_map_elewise_unary");  \
    }                                                                          \
  }

/**
 * @brief Generic compute elewise unary
 *
 * @param input_ptr: C pointer, input values
 * @param output_ptr: C pointer, output values
 * @param data_type: CTH_TENSOR_DATA_TYPE, input type
 * @param N: cth_tensor_dim_t, num of elements
 * @param kernel: compute function
 */
#define _cpu_1d_map_elewise_unary_generic(                                     \
    input_ptr, output_ptr, data_type, N, kernel)                               \
  {                                                                            \
    if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                              \
      kernel(input_ptr, output_ptr, N, bool);                                  \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      kernel(input_ptr, output_ptr, N, int16_t);                               \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      kernel(input_ptr, output_ptr, N, int32_t);                               \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      kernel(input_ptr, output_ptr, N, int64_t);                               \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      kernel(input_ptr, output_ptr, N, uint8_t);                               \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {                   \
      kernel(input_ptr, output_ptr, N, float);                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                   \
      kernel(input_ptr, output_ptr, N, float);                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      kernel(input_ptr, output_ptr, N, double);                                \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR,                                                         \
          "Unsupported data type in _cpu_1d_map_elewise_unary_generic");       \
    }                                                                          \
  }

/**
 * @brief Apply 1d binary kernel on input and output pointers in element wise
 * way
 *
 * @note This one assumes both input have same data types
 *
 * @param input_ptr_a: C pointer, input a
 * @param input_ptr_b: C pointer, input b
 * @param output_ptr: C pointer, output
 * @param data_type: CTH_TENSOR_DATA_TYPE, input type
 * @param N: cth_tensor_dim_t, num of elements
 * @param kernel: compute function
 */
#define _cpu_1d_map_elewise_binary(                                            \
    input_ptr_a, input_ptr_b, output_ptr, data_type, N, kernel)                \
  {                                                                            \
    if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                              \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, bool);              \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, int16_t);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, int32_t);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, int64_t);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, uint8_t);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {                   \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, float);             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                   \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, float);             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      _cpu_1d_map_elewise_binary_forloop(                                      \
          input_ptr_a, input_ptr_b, output_ptr, N, kernel, double);            \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR, "Unsupported data type in _cpu_1d_map_elewise_binary"); \
    }                                                                          \
  }

/**
 * @brief Expand a computation to a generic type
 *
 * @param op: operator
 * @param compute_fn: compute function
 * @param data_type: CTH_TENSOR_DATA_TYPE, inpute type
 */
#define _cpu_generic_compute(op, compute_fn, data_type)                        \
  {                                                                            \
    if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                              \
      compute_fn(op, bool);                                                    \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      compute_fn(op, int16_t);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      compute_fn(op, int32_t);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      compute_fn(op, int64_t);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      compute_fn(op, uint8_t);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {                   \
      compute_fn(op, float);                                                   \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                   \
      compute_fn(op, float);                                                   \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      compute_fn(op, double);                                                  \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type in _cpu_generic_compute"); \
    }                                                                          \
  }

/**
 * @brief Expand a computation to a generic type with input, output type
 *
 * @param op: operator
 * @param compute_fn: compute function
 * @param input_type: CTH_TENSOR_DATA_TYPE, input type
 * @param output_type: CTH_TENSOR_DATA_TYPE, output type
 */
#define _cpu_generic_compute2(op, compute_fn, input_type, output_type)         \
  {                                                                            \
    if (input_type == CTH_TENSOR_DATA_TYPE_BOOL) {                             \
      compute_fn(op, bool, input_type, output_type);                           \
    } else if (input_type == CTH_TENSOR_DATA_TYPE_INT_16) {                    \
      compute_fn(op, int16_t, input_type, output_type);                        \
    } else if (input_type == CTH_TENSOR_DATA_TYPE_INT_32) {                    \
      compute_fn(op, int32_t, input_type, output_type);                        \
    } else if (input_type == CTH_TENSOR_DATA_TYPE_INT_64) {                    \
      compute_fn(op, int64_t, input_type, output_type);                        \
    } else if (input_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                    \
      compute_fn(op, uint8_t, input_type, output_type);                        \
    } else if (input_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {                  \
      compute_fn(op, float, input_type, output_type);                          \
    } else if (input_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                  \
      compute_fn(op, float, input_type, output_type);                          \
    } else if (input_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                  \
      compute_fn(op, double, input_type, output_type);                         \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type in _cpu_generic_compute"); \
    }                                                                          \
  }

/**
 * @brief Expand a computation to all bit computation supported types
 *
 * @param op: operator
 * @param compute_fn: compute function
 * @param data_type: CTH_TENSOR_DATA_TYPE, input data type
 */
#define _cpu_bit_compute(op, compute_fn, data_type)                            \
  {                                                                            \
    if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                              \
      compute_fn(op, bool);                                                    \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      compute_fn(op, int16_t);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      compute_fn(op, int32_t);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      compute_fn(op, int64_t);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      compute_fn(op, uint8_t);                                                 \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type in _cpu_bit_compute");     \
    }                                                                          \
  }

/**
 * @brief General reduce computation
 *
 * @param op: operator
 * @param reduce_action: reduce function
 * @param input_data_type: input data C type
 * @param output_data_type: output data C type
 * @param input_dtype_enum: CTH_TENSOR_DATA_TYPE, input data type
 * @param output_dtype_enum: CTH_TENSOR_DATA_TYPE, output data type
 *
 */
#define _cpu_reduce_dim_compute(                                               \
    op,                                                                        \
    reduce_action,                                                             \
    input_data_type,                                                           \
    output_data_type,                                                          \
    input_dtype_enum,                                                          \
    output_dtype_enum)                                                         \
  do {                                                                         \
    CTHParam *dim_param = cth_get_param_by_type(op, CTH_PARAM_TYPE_DIM, true); \
    cth_tensor_dim_t reduce_dim = *(dim_param->data.dim_val);                  \
    cth_tensor_dim_t reduce_dim_size;                                          \
    if (reduce_dim == -1) {                                                    \
      reduce_dim_size = in->meta_info->n_elements;                             \
    } else {                                                                   \
      reduce_dim_size = in->meta_info->dims[reduce_dim];                       \
    }                                                                          \
                                                                               \
    CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);          \
    CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);        \
    input_data_type *in_ptr = (input_data_type *)in->values;                   \
    output_data_type *out_ptr = (output_data_type *)out->values;               \
                                                                               \
    cth_tensor_dim_t input_n_eles = in->meta_info->n_elements;                 \
    cth_tensor_dim_t tensor_n_dim = in->meta_info->n_dim;                      \
    cth_tensor_dim_t num_reduce_groups = input_n_eles / reduce_dim_size;       \
    cth_tensor_dim_t inner_offset =                                            \
        cth_tensor_reduce_inneroffset(in, reduce_dim);                         \
    cth_tensor_dim_t reduce_index_size = tensor_n_dim - 1;                     \
    if (reduce_dim == -1) {                                                    \
      reduce_index_size = 1;                                                   \
    }                                                                          \
    cth_tensor_dim_t *reduce_index_dims =                                      \
        MALLOC(sizeof(cth_tensor_dim_t) * reduce_index_size);                  \
    for (cth_tensor_dim_t group_i = 0; group_i < num_reduce_groups;            \
         group_i++) {                                                          \
      cth_tensor_get_reduce_index(in, group_i, reduce_dim, reduce_index_dims); \
      cth_tensor_dim_t start_offset =                                          \
          cth_tensor_reduce_startoffset(in, reduce_index_dims, reduce_dim);    \
      reduce_action(                                                           \
          in_ptr,                                                              \
          out_ptr,                                                             \
          input_data_type,                                                     \
          output_data_type,                                                    \
          input_dtype_enum,                                                    \
          output_dtype_enum,                                                   \
          start_offset,                                                        \
          inner_offset,                                                        \
          group_i,                                                             \
          reduce_dim_size);                                                    \
    }                                                                          \
    FREE(reduce_index_dims);                                                   \
  } while (0)

/**
 * @brief convert output type
 *
 * @param op: operator
 * @param reduce_action: reduce function
 * @param input_t: input data C type
 * @param output_data_type: CTH_TENSOR_DATA_TYPE, output data type
 * @param input_data_type: CTH_TENSOR_DATA_TYPE, input data type
 */
#define _cput_reduce_dim_generic_out_type(                                     \
    op, reduce_action, input_t, output_data_type, input_data_type)             \
  do {                                                                         \
    if (output_data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                       \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          bool,                                                                \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else if (output_data_type == CTH_TENSOR_DATA_TYPE_INT_16) {              \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          int16_t,                                                             \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else if (output_data_type == CTH_TENSOR_DATA_TYPE_INT_32) {              \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          int32_t,                                                             \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else if (output_data_type == CTH_TENSOR_DATA_TYPE_INT_64) {              \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          int64_t,                                                             \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else if (output_data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {              \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          uint8_t,                                                             \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else if (output_data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {            \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          float,                                                               \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else if (output_data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {            \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          float,                                                               \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else if (output_data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {            \
      _cpu_reduce_dim_compute(                                                 \
          op,                                                                  \
          reduce_action,                                                       \
          input_t,                                                             \
          double,                                                              \
          input_data_type,                                                     \
          output_data_type);                                                   \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type in _cpu_generic_compute"); \
    }                                                                          \
  } while (0)

/**
 * @brief General logic to do reduce op along a dim
 *
 * @param op: operator
 * @param input_data_type: CTH_TENSOR_DATA_TYPE, input data type
 * @param out_data_type: CTH_TENSOR_DATA_TYPE, output data type
 * @param reduce_action: reduce function
 */
#define _cpu_reduce_dim_generic(                                               \
    op, input_data_type, output_data_type, reduce_action)                      \
  do {                                                                         \
    if (input_data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                        \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, bool, output_data_type, input_data_type);         \
    } else if (input_data_type == CTH_TENSOR_DATA_TYPE_INT_16) {               \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, int16_t, output_data_type, input_data_type);      \
    } else if (input_data_type == CTH_TENSOR_DATA_TYPE_INT_32) {               \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, int32_t, output_data_type, input_data_type);      \
    } else if (input_data_type == CTH_TENSOR_DATA_TYPE_INT_64) {               \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, int64_t, output_data_type, input_data_type);      \
    } else if (input_data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {               \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, uint8_t, output_data_type, input_data_type);      \
    } else if (input_data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {             \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, float, output_data_type, input_data_type);        \
    } else if (input_data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {             \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, float, output_data_type, input_data_type);        \
    } else if (input_data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {             \
      _cput_reduce_dim_generic_out_type(                                       \
          op, reduce_action, double, output_data_type, input_data_type);       \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type in _cpu_generic_compute"); \
    }                                                                          \
  } while (0)

/**
 * @brief Sort an array inplace
 *
 * @param array: array pointer
 * @param data_type: C type
 * @param offset: int, offset between each element in array_ptr
 * @param size: cth_array_index_t, array size
 */
#define _cth_sort_array_inplace(array, data_type, offset, size)                \
  do {                                                                         \
                                                                               \
  } while (0)

/**
 * @brief Sort an array in place with index tracking
 *
 * @param array: array pointer
 * @param index: index pointer
 * @param data_type: C type
 * @param offset: int, offset between each element in array_ptr
 * @param size: cth_array_index_t, array_size
 */
#define _cth_sort_array_inplace_tracking(                                      \
    array, index, data_type, offset, size)                                     \
  do {                                                                         \
    _cth_sort_array_inplace(array, data_type, offset, size);                   \
                                                                               \
    for (cth_array_index_t i = 0; i < size; i++) {                             \
      index[i] = i;                                                            \
    }                                                                          \
                                                                               \
  } while (0)

/**
 * @brief The real padding logic flow. The flow copy original content to output
 * tensor and let the `padding_logic` to handle the padding content.
 *
 * @param op CTHOperator op
 * @param data_type C data type
 * @param padding_logic padding logic
 * @param extra_force extra force checking logic
 *
 */
#define _cth_padding_flow_1d(op, data_type, padding_logic, extra_force)        \
  do {                                                                         \
    CTHTensor *in_tensor = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);   \
    CTHTensor *out_tensor = cth_array_at(CTHTensor)(op->out_bound_tensors, 0); \
    CTHTensorMeta *in_meta = in_tensor->meta_info;                             \
    CTHTensorMeta *out_meta = out_tensor->meta_info;                           \
    data_type *in_ptr = (data_type *)in_tensor->values;                        \
    data_type *out_ptr = (data_type *)out_tensor->values;                      \
    CTHDim2 *padding;                                                          \
    cth_extract_param_value(                                                   \
        op, CTH_PARAM_TYPE_PADDING_D2, (void **)&padding, true);               \
    cth_tensor_dim_t padding_left = padding->d_0;                              \
    cth_tensor_dim_t padding_right = padding->d_1;                             \
    cth_tensor_dim_t input_b_dim = in_meta->dims[0];                           \
    cth_tensor_dim_t input_z_dim = in_meta->dims[1];                           \
    cth_tensor_dim_t input_x_dim = in_meta->dims[2];                           \
    cth_tensor_dim_t output_b_dim = out_meta->dims[0];                         \
    cth_tensor_dim_t output_z_dim = out_meta->dims[1];                         \
    cth_tensor_dim_t output_x_dim = out_meta->dims[2];                         \
                                                                               \
    FORCE_EQ(                                                                  \
        input_b_dim,                                                           \
        output_b_dim,                                                          \
        "_cth_padding_flow_1d dimension check fails: batch size. Input: %ld, " \
        "output: %ld",                                                         \
        input_b_dim,                                                           \
        output_b_dim);                                                         \
    FORCE_EQ(                                                                  \
        input_z_dim,                                                           \
        output_z_dim,                                                          \
        "_cth_padding_flow_1d dimension check fails: z. Input: %ld, output: "  \
        "%ld",                                                                 \
        input_z_dim,                                                           \
        output_z_dim);                                                         \
    FORCE_EQ(                                                                  \
        input_x_dim + padding_left + padding_right,                            \
        output_x_dim,                                                          \
        "_cth_padding_flow_1d dimension check fails: x. Input: %ld, output: "  \
        "%ld",                                                                 \
        input_x_dim,                                                           \
        output_x_dim);                                                         \
                                                                               \
    extra_force();                                                             \
                                                                               \
    for (cth_tensor_dim_t batch_i = 0; batch_i < output_b_dim; batch_i++) {    \
      for (cth_tensor_dim_t z_i = 0; z_i < output_z_dim; z_i++) {              \
        cth_tensor_dim_t in_offset =                                           \
            batch_i * (input_z_dim * input_x_dim) + z_i * (input_x_dim);       \
        cth_tensor_dim_t out_offset =                                          \
            batch_i * (output_z_dim * output_x_dim) + z_i * (output_x_dim);    \
        memcpy(                                                                \
            out_ptr + out_offset + padding_left,                               \
            in_ptr + in_offset,                                                \
            sizeof(data_type) * input_x_dim);                                  \
                                                                               \
        padding_logic(op, data_type);                                          \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Generic 1D padding logic. It basically just dispatch by data type.
 *
 * @param op CTHOperator op
 * @param padding_logic the macro to implement a specific padding
 * @param extra_force extra force checking logic
 */
#define _cth_padding_generic_1d(op, padding_logic, extra_force)                \
  do {                                                                         \
    CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);          \
    CTH_TENSOR_DATA_TYPE data_type_enum = in->meta_info->data_type;            \
    if (data_type_enum == CTH_TENSOR_DATA_TYPE_BOOL) {                         \
      _cth_padding_flow_1d(op, bool, padding_logic, extra_force);              \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_INT_16) {                \
      _cth_padding_flow_1d(op, int16_t, padding_logic, extra_force);           \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_INT_32) {                \
      _cth_padding_flow_1d(op, int32_t, padding_logic, extra_force);           \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_INT_64) {                \
      _cth_padding_flow_1d(op, int64_t, padding_logic, extra_force);           \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_UINT_8) {                \
      _cth_padding_flow_1d(op, uint8_t, padding_logic, extra_force);           \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_FLOAT_16) {              \
      _cth_padding_flow_1d(op, float, padding_logic, extra_force);             \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_FLOAT_32) {              \
      _cth_padding_flow_1d(op, float, padding_logic, extra_force);             \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_FLOAT_64) {              \
      _cth_padding_flow_1d(op, double, padding_logic, extra_force);            \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR, "Unsupported data type in _cth_padding_generic_1d");    \
    }                                                                          \
  } while (0)

/**
 * @brief The real padding logic flow. The flow copy original content to output
 * tensor and let the `padding_logic` to handle the padding content.
 *
 * @param op CTHOperator op
 * @param data_type C data type
 * @param padding_logic the macro to implement a specific padding
 * @param extra_force extra force logic
 */
#define _cth_padding_flow_2d(op, data_type, padding_logic, extra_force)        \
  do {                                                                         \
    CTHTensor *in_tensor = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);   \
    CTHTensor *out_tensor = cth_array_at(CTHTensor)(op->out_bound_tensors, 0); \
    CTHTensorMeta *in_meta = in_tensor->meta_info;                             \
    CTHTensorMeta *out_meta = out_tensor->meta_info;                           \
    data_type *in_ptr = (data_type *)in_tensor->values;                        \
    data_type *out_ptr = (data_type *)out_tensor->values;                      \
    CTHDim4 *padding;                                                          \
    cth_extract_param_value(                                                   \
        op, CTH_PARAM_TYPE_PADDING_D4, (void **)&padding, true);               \
    cth_tensor_dim_t padding_left = padding->d_0;                              \
    cth_tensor_dim_t padding_right = padding->d_1;                             \
    cth_tensor_dim_t padding_top = padding->d_2;                               \
    cth_tensor_dim_t padding_bottom = padding->d_3;                            \
    cth_tensor_dim_t in_x_dim = in_meta->dims[3];                              \
    cth_tensor_dim_t in_y_dim = in_meta->dims[2];                              \
    cth_tensor_dim_t in_c_dim = in_meta->dims[1];                              \
    cth_tensor_dim_t in_b_dim = in_meta->dims[0];                              \
    cth_tensor_dim_t out_x_dim = out_meta->dims[3];                            \
    cth_tensor_dim_t out_y_dim = out_meta->dims[2];                            \
    cth_tensor_dim_t out_c_dim = out_meta->dims[1];                            \
    cth_tensor_dim_t out_b_dim = out_meta->dims[0];                            \
                                                                               \
    FORCE_EQ(                                                                  \
        in_b_dim,                                                              \
        out_b_dim,                                                             \
        "_cth_padding_flow_2d dimension check fails: batch size. Input: "      \
        "%ld, "                                                                \
        "output: %ld",                                                         \
        in_b_dim,                                                              \
        out_b_dim);                                                            \
    FORCE_EQ(                                                                  \
        in_c_dim,                                                              \
        out_c_dim,                                                             \
        "_cth_padding_flow_2d dimension check fails: channel. Input: %ld, "    \
        "output: "                                                             \
        "%ld",                                                                 \
        in_c_dim,                                                              \
        out_c_dim);                                                            \
    FORCE_EQ(                                                                  \
        in_y_dim + padding_top + padding_bottom,                               \
        out_y_dim,                                                             \
        "_cth_padding_flow_2d dimension check fails: y. Input: %ld, output: "  \
        "%ld",                                                                 \
        in_y_dim,                                                              \
        out_y_dim);                                                            \
    FORCE_EQ(                                                                  \
        in_x_dim + padding_left + padding_right,                               \
        out_x_dim,                                                             \
        "_cth_padding_flow_2d dimension check fails: x. Input: %ld, output: "  \
        "%ld",                                                                 \
        in_x_dim,                                                              \
        out_x_dim);                                                            \
    extra_force();                                                             \
                                                                               \
    for (cth_tensor_dim_t b_i = 0; b_i < out_meta->dims[0]; b_i++) {           \
      for (cth_tensor_dim_t c_i = 0; c_i < out_meta->dims[1]; c_i++) {         \
        for (cth_tensor_dim_t y_i = 0; y_i < out_meta->dims[2]; y_i++) {       \
          cth_tensor_dim_t out_offset =                                        \
              b_i * (out_c_dim * out_y_dim * out_x_dim) +                      \
              c_i * (out_y_dim * out_x_dim) + y_i * (out_x_dim);               \
                                                                               \
          cth_tensor_dim_t in_offset;                                          \
          bool padding_whole_row = false;                                      \
          if (y_i < padding_top) {                                             \
            in_offset = b_i * (in_c_dim * in_y_dim * in_x_dim) +               \
                        c_i * (in_y_dim * in_x_dim);                           \
            padding_whole_row = true;                                          \
          } else if (y_i >= (padding_top + in_y_dim)) {                        \
            in_offset = b_i * (in_c_dim * in_y_dim * in_x_dim) +               \
                        c_i * (in_y_dim * in_x_dim) +                          \
                        (in_y_dim - 1) * in_x_dim;                             \
            padding_whole_row = true;                                          \
          } else {                                                             \
            in_offset = b_i * (in_c_dim * in_y_dim * in_x_dim) +               \
                        c_i * (in_y_dim * in_x_dim) +                          \
                        (y_i - padding_top) * in_x_dim;                        \
            memcpy(                                                            \
                out_ptr + out_offset + padding_left,                           \
                in_ptr + in_offset,                                            \
                sizeof(data_type) * in_x_dim);                                 \
          }                                                                    \
                                                                               \
          padding_logic(op, data_type);                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Generic 2D padding logic. It basically just dispatch by data type.
 *
 * @param op CTHOperator op
 * @param padding_logic the macro to implement a specific padding
 * @param extra_force extra force logic
 */
#define _cth_padding_generic_2d(op, padding_logic, extra_force)                \
  do {                                                                         \
    CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);          \
    CTH_TENSOR_DATA_TYPE data_type = in->meta_info->data_type;                 \
    if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                              \
      _cth_padding_flow_2d(op, bool, padding_logic, extra_force);              \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      _cth_padding_flow_2d(op, int16_t, padding_logic, extra_force);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      _cth_padding_flow_2d(op, int32_t, padding_logic, extra_force);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      _cth_padding_flow_2d(op, int64_t, padding_logic, extra_force);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      _cth_padding_flow_2d(op, uint8_t, padding_logic, extra_force);           \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {                   \
      _cth_padding_flow_2d(op, float, padding_logic, extra_force);             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                   \
      _cth_padding_flow_2d(op, float, padding_logic, extra_force);             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      _cth_padding_flow_2d(op, double, padding_logic, extra_force);            \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR, "Unsupported data type in _cth_padding_generic_2d");    \
    }                                                                          \
  } while (0)

/**
 * @brief The real padding logic flow. The flow copy original content to output
 * tensor and let the `padding_logic` to handle the padding content.
 *
 * The flow will first fill unpadded part. Then fill purelly padded part (front,
 * back)
 *
 * @param op CTHOperator op
 * @param data_type C data type
 * @param padding_logic the macro to implement a specific padding for regular
 * frame
 * @param padding_logic_front the macro to padding front part
 * @param padding_logic_back  the macro to padding back part
 *
 */
#define _cth_padding_flow_3d(                                                  \
    op, data_type, padding_logic, padding_logic_front, padding_logic_back)     \
  do {                                                                         \
    CTHTensor *in_tensor = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);   \
    CTHTensor *out_tensor = cth_array_at(CTHTensor)(op->out_bound_tensors, 0); \
    CTHTensorMeta *in_meta = in_tensor->meta_info;                             \
    CTHTensorMeta *out_meta = out_tensor->meta_info;                           \
    data_type *in_ptr = (data_type *)in_tensor->values;                        \
    data_type *out_ptr = (data_type *)out_tensor->values;                      \
    size_t data_size = sizeof(data_type);                                      \
    CTHDim6 *padding;                                                          \
    cth_extract_param_value(                                                   \
        op, CTH_PARAM_TYPE_PADDING_D6, (void **)&padding, true);               \
    cth_tensor_dim_t padding_left = padding->d_0;                              \
    cth_tensor_dim_t padding_right = padding->d_1;                             \
    cth_tensor_dim_t padding_top = padding->d_2;                               \
    cth_tensor_dim_t padding_bottom = padding->d_3;                            \
    cth_tensor_dim_t padding_front = padding->d_4;                             \
    cth_tensor_dim_t padding_back = padding->d_5;                              \
    cth_tensor_dim_t in_x_dim = in_meta->dims[4];                              \
    cth_tensor_dim_t in_y_dim = in_meta->dims[3];                              \
    cth_tensor_dim_t in_z_dim = in_meta->dims[2];                              \
    cth_tensor_dim_t in_c_dim = in_meta->dims[1];                              \
    cth_tensor_dim_t in_b_dim = in_meta->dims[0];                              \
    cth_tensor_dim_t out_x_dim = out_meta->dims[4];                            \
    cth_tensor_dim_t out_y_dim = out_meta->dims[3];                            \
    cth_tensor_dim_t out_z_dim = out_meta->dims[2];                            \
    cth_tensor_dim_t out_c_dim = out_meta->dims[1];                            \
    cth_tensor_dim_t out_b_dim = out_meta->dims[0];                            \
                                                                               \
    FORCE_EQ(                                                                  \
        in_b_dim,                                                              \
        out_b_dim,                                                             \
        "_cth_padding_flow_3d dimension check fails: batch size. Input: "      \
        "%ld, "                                                                \
        "output: %ld",                                                         \
        in_b_dim,                                                              \
        out_b_dim);                                                            \
    FORCE_EQ(                                                                  \
        in_c_dim,                                                              \
        out_c_dim,                                                             \
        "_cth_padding_flow_3d dimension check fails: channel. Input: %ld, "    \
        "output: "                                                             \
        "%ld",                                                                 \
        in_c_dim,                                                              \
        out_c_dim);                                                            \
    FORCE_EQ(                                                                  \
        in_z_dim + padding_front + padding_back,                               \
        out_z_dim,                                                             \
        "_cth_padding_flow_3d dimension check fails: z. Input: %ld, output: "  \
        "%ld",                                                                 \
        in_z_dim,                                                              \
        out_z_dim);                                                            \
    FORCE_EQ(                                                                  \
        in_y_dim + padding_top + padding_bottom,                               \
        out_y_dim,                                                             \
        "_cth_padding_flow_3d dimension check fails: y. Input: %ld, output: "  \
        "%ld",                                                                 \
        in_y_dim,                                                              \
        out_y_dim);                                                            \
    FORCE_EQ(                                                                  \
        in_x_dim + padding_left + padding_right,                               \
        out_x_dim,                                                             \
        "_cth_padding_flow_3d dimension check fails: x. Input: %ld, output: "  \
        "%ld",                                                                 \
        in_x_dim,                                                              \
        out_x_dim);                                                            \
                                                                               \
    for (cth_tensor_dim_t b_i = 0; b_i < out_b_dim; b_i++) {                   \
      for (cth_tensor_dim_t c_i = 0; c_i < out_c_dim; c_i++) {                 \
        for (cth_tensor_dim_t z_i = padding_front;                             \
             z_i < (out_z_dim - padding_back);                                 \
             z_i++) {                                                          \
          for (cth_tensor_dim_t y_i = 0; y_i < out_y_dim; y_i++) {             \
            cth_tensor_dim_t out_offset =                                      \
                b_i * (out_c_dim * out_z_dim * out_y_dim * out_x_dim) +        \
                c_i * (out_z_dim * out_y_dim * out_x_dim) +                    \
                z_i * (out_y_dim * out_x_dim) + y_i * (out_x_dim);             \
            bool padding_whole_row = false;                                    \
            cth_tensor_dim_t in_offset = 0;                                    \
                                                                               \
            if (y_i < padding_top) {                                           \
              in_offset = b_i * (in_c_dim * in_z_dim * in_y_dim * in_x_dim) +  \
                          c_i * (in_z_dim * in_y_dim * in_x_dim) +             \
                          (z_i - padding_front) * (in_y_dim * in_x_dim) +      \
                          0 * in_x_dim;                                        \
              padding_whole_row = true;                                        \
            } else if (y_i >= (padding_top + in_y_dim)) {                      \
              in_offset = b_i * (in_c_dim * in_z_dim * in_y_dim * in_x_dim) +  \
                          c_i * (in_z_dim * in_y_dim * in_x_dim) +             \
                          (z_i - padding_front) * (in_y_dim * in_x_dim) +      \
                          (in_y_dim - 1) * in_x_dim;                           \
              padding_whole_row = true;                                        \
            } else {                                                           \
              in_offset = b_i * (in_c_dim * in_z_dim * in_y_dim * in_x_dim) +  \
                          c_i * (in_z_dim * in_y_dim * in_x_dim) +             \
                          (z_i - padding_front) * (in_y_dim * in_x_dim) +      \
                          (y_i - padding_top) * in_x_dim;                      \
              memcpy(                                                          \
                  out_ptr + out_offset + padding_left,                         \
                  in_ptr + in_offset,                                          \
                  sizeof(data_type) * in_x_dim);                               \
            }                                                                  \
                                                                               \
            padding_logic(op, data_type);                                      \
          }                                                                    \
        }                                                                      \
                                                                               \
        if (padding_front > 0) {                                               \
          padding_logic_front(op, data_type);                                  \
        }                                                                      \
                                                                               \
        if (padding_back > 0) {                                                \
          padding_logic_back(op, data_type);                                   \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Generic 3D padding logic. It basically just dispatch by data type.
 *
 * @param op CTHOperator op
 * @param padding_logic the macro to implement a specific padding
 */
#define _cth_padding_generic_3d(                                               \
    op, padding_logic, padding_logic_front, padding_logic_back)                \
  do {                                                                         \
    CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);          \
    CTH_TENSOR_DATA_TYPE data_type_enum = in->meta_info->data_type;            \
    if (data_type_enum == CTH_TENSOR_DATA_TYPE_BOOL) {                         \
      _cth_padding_flow_3d(                                                    \
          op, bool, padding_logic, padding_logic_front, padding_logic_back);   \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_INT_16) {                \
      _cth_padding_flow_3d(                                                    \
          op,                                                                  \
          int16_t,                                                             \
          padding_logic,                                                       \
          padding_logic_front,                                                 \
          padding_logic_back);                                                 \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_INT_32) {                \
      _cth_padding_flow_3d(                                                    \
          op,                                                                  \
          int32_t,                                                             \
          padding_logic,                                                       \
          padding_logic_front,                                                 \
          padding_logic_back);                                                 \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_INT_64) {                \
      _cth_padding_flow_3d(                                                    \
          op,                                                                  \
          int64_t,                                                             \
          padding_logic,                                                       \
          padding_logic_front,                                                 \
          padding_logic_back);                                                 \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_UINT_8) {                \
      _cth_padding_flow_3d(                                                    \
          op,                                                                  \
          uint8_t,                                                             \
          padding_logic,                                                       \
          padding_logic_front,                                                 \
          padding_logic_back);                                                 \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_FLOAT_16) {              \
      _cth_padding_flow_3d(                                                    \
          op, float, padding_logic, padding_logic_front, padding_logic_back);  \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_FLOAT_32) {              \
      _cth_padding_flow_3d(                                                    \
          op, float, padding_logic, padding_logic_front, padding_logic_back);  \
    } else if (data_type_enum == CTH_TENSOR_DATA_TYPE_FLOAT_64) {              \
      _cth_padding_flow_3d(                                                    \
          op, double, padding_logic, padding_logic_front, padding_logic_back); \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR, "Unsupported data type in _cth_padding_generic_3d");    \
    }                                                                          \
  } while (0)

#endif /* X86_COMMON_H */
