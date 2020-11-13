#ifndef CTH_DEFAULT_UTIL_H
#define CTH_DEFAULT_UTIL_H

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
    cth_tensor_dim_t reduce_dim = (cth_tensor_dim_t) * (dim_param->data.dim);  \
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

#endif /* X86_COMMON_H */
