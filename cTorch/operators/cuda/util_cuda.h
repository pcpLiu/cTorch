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

#ifndef CTH_UTIL_CUDA_H
#define CTH_UTIL_CUDA_H

/**
 * @brief thread per block
 */
#define CTH_CUDA_THREADS_PER_BLOCK 1024

/**
 * @brief Kernel function name for float
 */
#define _cth_cuda_kernel_func_float(kernel_func_name) kernel_func_name##_f

/**
 * @brief Kernel function name for double
 */
#define _cth_cuda_kernel_func_double(kernel_func_name) kernel_func_name##_d

/**
 * @brief Layout two kernel functions based on kernel func for float and double
 *
 * This will layout two kernel functions:
 *    - __global__ void xxx##f(float* in_ptr_d, float* out_ptr_d, int N){...}
 *    - __global__ void xxx##d(double* in_ptr_d, double* out_ptr_d, int N){...}
 *
 */
#define _cth_declare_cuda_unary_kernel(kernel_func_name, f_func, d_func)       \
  _cth_declare_cuda_unary_kernel_type(                                         \
      _cth_cuda_kernel_func_float(kernel_func_name), float, f_func);           \
  _cth_declare_cuda_unary_kernel_type(                                         \
      _cth_cuda_kernel_func_double(kernel_func_name), double, d_func)

/**
 * @brief Implement a unary cuda kernel function
 *
 */
#define _cth_declare_cuda_unary_kernel_type(kernel_func_name, data_type, func) \
  __global__ void kernel_func_name(                                            \
      data_type *in_ptr_d, data_type *out_ptr_d, int N) {                      \
    int i = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (i < N) {                                                               \
      out_ptr_d[i] = func(in_ptr_d[i]);                                        \
    }                                                                          \
  }

/**
 * @brief Layout two kernel functions based on kernel func for float and double
 *
 * This will layout two kernel functions:
 *    - __global__ void xxx##f(float* in_ptr_1_d, float* in_ptr_2_d, float*
 * out_ptr_d, int N){...}
 *    - __global__ void xxx##d(double* in_ptr_1_d, double* in_ptr_2_d, double*
 * out_ptr_d, int N){...}
 *
 */
#define _cth_declare_cuda_binary_kernel(kernel_func_name, f_func, d_func)      \
  _cth_declare_cuda_binary_kernel_type(                                        \
      _cth_cuda_kernel_func_float(kernel_func_name), float, f_func);           \
  _cth_declare_cuda_binary_kernel_type(                                        \
      _cth_cuda_kernel_func_double(kernel_func_name), double, d_func)

/**
 * @brief Implement a bina cuda kernel function
 *
 */
#define _cth_declare_cuda_binary_kernel_type(                                  \
    kernel_func_name, data_type, func)                                         \
  __global__ void kernel_func_name(                                            \
      data_type *in_ptr_1_d,                                                   \
      data_type *in_ptr_2_d,                                                   \
      data_type *out_ptr_d,                                                    \
      int N) {                                                                 \
    int i = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (i < N) {                                                               \
      out_ptr_d[i] = func(in_ptr_1_d[i], in_ptr_2_d[i]);                       \
    }                                                                          \
  }

/**
 * @brief Layout two kernel functions based on kernel func for float and double
 *
 * This will layout two kernel functions:
 *    - __global__ void xxx##f(float* in_ptr_1_d, float* in_ptr_2_d, float*
 * out_ptr_d, int N){...}
 *    - __global__ void xxx##d(double* in_ptr_1_d, double* in_ptr_2_d, double*
 * out_ptr_d, int N){...}
 *
 */
#define _cth_declare_cuda_binary_kernel_generic(                               \
    kernel_func_name, f_block, d_block)                                        \
  _cth_declare_cuda_binary_kernel_type_generic(                                \
      _cth_cuda_kernel_func_float(kernel_func_name), float, f_block);          \
  _cth_declare_cuda_binary_kernel_type_generic(                                \
      _cth_cuda_kernel_func_double(kernel_func_name), double, d_block)

/**
 * @brief Implement a bina cuda kernel function with generic block
 *
 */
#define _cth_declare_cuda_binary_kernel_type_generic(                          \
    kernel_func_name, data_type, generic_block)                                \
  __global__ void kernel_func_name(                                            \
      data_type *in_ptr_1_d,                                                   \
      data_type *in_ptr_2_d,                                                   \
      data_type *out_ptr_d,                                                    \
      int N) {                                                                 \
    int i = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (i < N) {                                                               \
      generic_block(in_ptr_1_d, in_ptr_2_d, out_ptr_d, i);                     \
    }                                                                          \
  }

/**
 * @brief Unary operator working block
 *
 */
#define _cth_cuda_unary_block(                                                 \
    data_type, in_ptr, out_ptr, N, kernel, threads_per_block, device)          \
  do {                                                                         \
    int size = N * sizeof(data_type);                                          \
    data_type *in_ptr_d;                                                       \
    data_type *out_ptr_d;                                                      \
    if (device == CTH_TENSOR_DEVICE_NORMAL) {                                  \
      cudaMalloc(&in_ptr_d, size);                                             \
      cudaMalloc(&out_ptr_d, size);                                            \
      cudaMemcpy(in_ptr_d, in_ptr, size, cudaMemcpyHostToDevice);              \
      cudaMemcpy(out_ptr_d, out_ptr, size, cudaMemcpyHostToDevice);            \
    } else {                                                                   \
      in_ptr_d = (data_type *)in_ptr;                                          \
      out_ptr_d = (data_type *)out_ptr;                                        \
    }                                                                          \
                                                                               \
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;          \
    kernel<<<num_blocks, threads_per_block>>>(in_ptr_d, out_ptr_d, N);         \
                                                                               \
    if (device == CTH_TENSOR_DEVICE_NORMAL) {                                  \
      cudaMemcpy(out_ptr, out_ptr_d, size, cudaMemcpyDeviceToHost);            \
      cudaFree(in_ptr_d);                                                      \
      cudaFree(out_ptr_d);                                                     \
    }                                                                          \
  } while (0)

/**
 * @brief consturct a unary op workflow
 *
 */
#define _cth_cuda_unary_workflow(                                              \
    data_type,                                                                 \
    in_ptr,                                                                    \
    out_ptr,                                                                   \
    N,                                                                         \
    threads_per_block,                                                         \
    kernel_func_name,                                                          \
    device)                                                                    \
  do {                                                                         \
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                          \
      _cth_cuda_unary_block(                                                   \
          float,                                                               \
          in_ptr,                                                              \
          out_ptr,                                                             \
          N,                                                                   \
          _cth_cuda_kernel_func_float(kernel_func_name),                       \
          threads_per_block,                                                   \
          device);                                                             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      _cth_cuda_unary_block(                                                   \
          double,                                                              \
          in_ptr,                                                              \
          out_ptr,                                                             \
          N,                                                                   \
          _cth_cuda_kernel_func_double(kernel_func_name),                      \
          threads_per_block,                                                   \
          device);                                                             \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type on CUDA backend.");        \
    }                                                                          \
  } while (0)

/**
 * @brief Unary operator working block
 *
 */
#define _cth_cuda_binary_block(                                                \
    data_type,                                                                 \
    in_ptr_1,                                                                  \
    in_ptr_2,                                                                  \
    out_ptr,                                                                   \
    N,                                                                         \
    kernel,                                                                    \
    threads_per_block,                                                         \
    device)                                                                    \
  do {                                                                         \
    int size = N * sizeof(data_type);                                          \
    data_type *in_ptr_1_d;                                                     \
    data_type *in_ptr_2_d;                                                     \
    data_type *out_ptr_d;                                                      \
    if (device == CTH_TENSOR_DEVICE_NORMAL) {                                  \
      cudaMalloc(&in_ptr_1_d, size);                                           \
      cudaMalloc(&in_ptr_2_d, size);                                           \
      cudaMalloc(&out_ptr_d, size);                                            \
      cudaMemcpy(in_ptr_1_d, in_ptr_1, size, cudaMemcpyHostToDevice);          \
      cudaMemcpy(in_ptr_2_d, in_ptr_2, size, cudaMemcpyHostToDevice);          \
      cudaMemcpy(out_ptr_d, out_ptr, size, cudaMemcpyHostToDevice);            \
    } else {                                                                   \
      in_ptr_1_d = (data_type *)in_ptr_1;                                      \
      in_ptr_1_d = (data_type *)in_ptr_1;                                      \
      out_ptr_d = (data_type *)out_ptr;                                        \
    }                                                                          \
                                                                               \
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;          \
    kernel<<<num_blocks, threads_per_block>>>(                                 \
        in_ptr_1_d, in_ptr_2_d, out_ptr_d, N);                                 \
                                                                               \
    if (device == CTH_TENSOR_DEVICE_NORMAL) {                                  \
      cudaMemcpy(out_ptr, out_ptr_d, size, cudaMemcpyDeviceToHost);            \
      cudaFree(in_ptr_1_d);                                                    \
      cudaFree(in_ptr_2_d);                                                    \
      cudaFree(out_ptr_d);                                                     \
    }                                                                          \
  } while (0)

/**
 * @brief consturct a binary op workflow
 *
 */
#define _cth_cuda_binary_workflow(                                             \
    data_type,                                                                 \
    in_ptr_1,                                                                  \
    in_ptr_2,                                                                  \
    out_ptr,                                                                   \
    N,                                                                         \
    threads_per_block,                                                         \
    kernel_func_name,                                                          \
    device)                                                                    \
  do {                                                                         \
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {                          \
      _cth_cuda_binary_block(                                                  \
          float,                                                               \
          in_ptr_1,                                                            \
          in_ptr_2,                                                            \
          out_ptr,                                                             \
          N,                                                                   \
          _cth_cuda_kernel_func_float(kernel_func_name),                       \
          threads_per_block,                                                   \
          device);                                                             \
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {                   \
      _cth_cuda_binary_block(                                                  \
          double,                                                              \
          in_ptr_1,                                                            \
          in_ptr_2,                                                            \
          out_ptr,                                                             \
          N,                                                                   \
          _cth_cuda_kernel_func_double(kernel_func_name),                      \
          threads_per_block,                                                   \
          device);                                                             \
    } else {                                                                   \
      FAIL_EXIT(CTH_LOG_ERR, "Unsupported data type on CUDA backend.");        \
    }                                                                          \
  } while (0)

#endif /* UTIL_CUDA_H */
