#ifndef CTH_X86_COMMON_H
#define CTH_X86_COMMON_H

#define _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, type_t)          \
  {                                                                            \
    type_t *input_t = (type_t *)input_ptr;                                     \
    type_t *output_t = (type_t *)output_ptr;                                   \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = kernel(input_t[i]);                                        \
    }                                                                          \
  }

/*
  Apply 1D kernel on input and output pointers in element-wise way.
*/
#define x86_1d_map(input_ptr, output_ptr, data_type, N, kernel)                \
  {                                                                            \
    if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {                              \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, char);             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {                     \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, int16_t);          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {                     \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, int32_t);          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {                     \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, int64_t);          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {                     \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, uint8_t);          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16) {                   \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, float_t);          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                   \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, float_t);          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      _x86_1d_map_forloop(input_ptr, output_ptr, N, kernel, double_t);         \
    }                                                                          \
  }

#endif /* X86_COMMON_H */
