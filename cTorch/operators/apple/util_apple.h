

#define _cth_apple_vforce_function_call_unary(                                 \
    data_type, func_name, in_ptr, out_ptr, N_ptr)                              \
  do {                                                                         \
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                          \
      vv##func_name##f((float *)out_ptr, (float *)in_ptr, (const int *)N_ptr); \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      vv##func_name((double *)out_ptr, (double *)in_ptr, (const int *)N_ptr);  \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR, "Unsupported data type on Apple backend calling");      \
    }                                                                          \
  } while (0)

#define _cth_apple_vforce_function_call_binary(                                \
    data_type, func_name, in_ptr_1, in_ptr_2, out_ptr, N_ptr)                  \
  do {                                                                         \
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                          \
      vv##func_name##f(                                                        \
          (float *)out_ptr,                                                    \
          (float *)in_ptr_1,                                                   \
          (float *)in_ptr_2,                                                   \
          (const int *)N_ptr);                                                 \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      vv##func_name(                                                           \
          (double *)out_ptr,                                                   \
          (double *)in_ptr_1,                                                  \
          (double *)in_ptr_2,                                                  \
          (const int *)N_ptr);                                                 \
    } else {                                                                   \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR, "Unsupported data type on Apple backend calling");      \
    }                                                                          \
  } while (0)
