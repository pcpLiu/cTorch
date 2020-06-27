#ifndef CTH_DEFAULT_UTIL_H
#define CTH_DEFAULT_UTIL_H

/**
 * Conduct unary operation element wise
 */
#define _cpu_1d_map_elewise_unary_forloop(                                     \
    input_ptr, output_ptr, N, kernel, type_t)                                  \
  {                                                                            \
    type_t *input_t = (type_t *)input_ptr;                                     \
    type_t *output_t = (type_t *)output_ptr;                                   \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = kernel(input_t[i]);                                        \
    }                                                                          \
  }

/**
 * Conduct binary operation element wise
 */
#define _cpu_1d_map_elewise_binary_forloop(                                    \
    input_ptr_a, input_ptr_b, output_ptr, N, kernel, type_t)                   \
  {                                                                            \
    type_t *input_a_t = (type_t *)input_ptr_a;                                 \
    type_t *input_b_t = (type_t *)input_ptr_b;                                 \
    type_t *output_t = (type_t *)output_ptr;                                   \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = kernel(input_a_t[i], input_b_t[i]);                        \
    }                                                                          \
  }

/**
 * Apply 1D unary kernel on input and output pointers in element wise way
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
 * Generic compute elewise unary
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
 * Apply 1d binary kernel on input and output pointers in element wise way
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
 * Expand a computation to a generic type
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
 * Expand a computation to all bit computation supported types
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

#endif /* X86_COMMON_H */
