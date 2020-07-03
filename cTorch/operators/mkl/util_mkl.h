#ifndef CTH_UTIL_MKL_H
#define CTH_UTIL_MKL_H

/**
 * @brief Call Intel Vector function macro
 *
 * @par Name convention:
 * https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/vector-mathematical-functions/vm-naming-conventions.html
 */
#define _cth_mkl_vm_function_call_unary(                                       \
    data_type, func_name, in_ptr, out_ptr, N)                                  \
  do {                                                                         \
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                          \
      vs##func_name(N, (float *)in_ptr, (float *)out_ptr);                     \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      vd##func_name(N, (double *)in_ptr, (double *)out_ptr);                   \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type on MKL calling");          \
    }                                                                          \
  } while (0)

#define _cth_mkl_vm_function_call_binary(                                      \
    data_type, func_name, in_ptr_1, in_ptr_2, out_ptr, N)                      \
  do {                                                                         \
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                          \
      vs##func_name(                                                           \
          N, (float *)in_ptr_1, (float *)in_ptr_2, (float *)out_ptr);          \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      vd##func_name(                                                           \
          N, (double *)in_ptr_1, (double *)in_ptr_2, (double *)out_ptr);       \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type on MKL calling");          \
    }                                                                          \
  } while (0)

#endif /* UTIL_MKL_H */
