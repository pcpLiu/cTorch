/**
 * Copyright 2021 Zhonghao Liu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * @brief vector x vector
 */
#define _cth_matmul_1(op, data_type)                                           \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
                                                                               \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    data_type *out_ptr = (data_type *)output->values;                          \
                                                                               \
    cth_tensor_dim_t N = input_1->meta_info->n_elements;                       \
    out_ptr[0] = 0;                                                            \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      out_ptr[0] += in_ptr_1[i] * in_ptr_2[i];                                 \
    }                                                                          \
  } while (0)

/**
 * @brief Matrix x Matrix
 */
#define _cth_matmul_2(op, data_type)                                           \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
                                                                               \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    data_type *out_ptr = (data_type *)output->values;                          \
                                                                               \
    cth_tensor_dim_t rows_1 = input_1->meta_info->dims[0];                     \
    cth_tensor_dim_t cols_1 = input_1->meta_info->dims[1];                     \
                                                                               \
    cth_tensor_dim_t rows_2 = input_2->meta_info->dims[0];                     \
    cth_tensor_dim_t cols_2 = input_2->meta_info->dims[1];                     \
                                                                               \
    cth_tensor_dim_t rows_out = output->meta_info->dims[0];                    \
    cth_tensor_dim_t cols_out = output->meta_info->dims[1];                    \
                                                                               \
    cth_tensor_dim_t N = cols_1;                                               \
                                                                               \
    for (cth_tensor_dim_t row = 0; row < rows_out; row++) {                    \
      cth_tensor_dim_t offset_1 = row * cols_1;                                \
                                                                               \
      for (cth_tensor_dim_t col = 0; col < cols_out; col++) {                  \
        cth_tensor_dim_t offset_2 = col;                                       \
        cth_tensor_dim_t offset_out = row * cols_out + col;                    \
                                                                               \
        out_ptr[offset_out] = (data_type)0;                                    \
        for (cth_tensor_dim_t i = 0; i < N; i++) {                             \
          out_ptr[offset_out] +=                                               \
              in_ptr_1[offset_1 + i] * in_ptr_2[offset_2 + i * cols_2];        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

/**
 * @brief Matrix product of two tensors. The behavior depends on the
 * dimensionality of the tensors. Check PyTorch doc:
 * https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul
 *
 * @par Op requirement:
 *    - # of input tensors: 2
 *      - 0: input 1
 *      - 1: input 2
 *    - # of output tensors: 1
 */
void op_matmul_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);

  CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);

  cth_tensor_dim_t *input_1_dims = input_1->meta_info->dims;
  cth_tensor_dim_t *input_2_dims = input_2->meta_info->dims;

  cth_tensor_dim_t input_1_ndim = input_1->meta_info->n_dim;
  cth_tensor_dim_t input_2_ndim = input_2->meta_info->n_dim;

  if ((input_1_ndim * input_2_ndim) == 1) {
    // If both tensors are 1-dimensional, the dot product (scalar) is
    // returned.

    FORCE_EQ(
        input_1_dims[0],
        input_2_dims[0],
        "op_matmul_cpu dim error. input_1 ([%ld]) should have same dim with "
        "input_2 ([%ld]).",
        input_1_dims[0],
        input_2_dims[0]);

    _cpu_generic_compute(op, _cth_matmul_1, input_1->meta_info->data_type);
  } else if (input_1_ndim == 2 && input_2_ndim == 2) {
    // If both arguments are 2-dimensional, the matrix-matrix product is
    // returned.

    FORCE_EQ(
        input_1_dims[1],
        input_2_dims[0],
        "op_matmul_cpu dim error. input_1 ([%ld, %ld]) and input_2 ([%ld, "
        "%ld]) cannot conduct matrix-matrix multiply.",
        input_1_dims[0],
        input_1_dims[1],
        input_2_dims[0],
        input_2_dims[1]);

    _cpu_generic_compute(op, _cth_matmul_2, input_1->meta_info->data_type);
  } else if (input_1_ndim == 1 && input_2_ndim == 2) {
    // If the first argument is 1-dimensional and the second argument is
    // 2-dimensional, a 1 is prepended to its dimension for the purpose of the
    // matrix multiply. After the matrix multiply, the prepended dimension is
    // removed.

    FORCE_EQ(
        input_1_dims[0],
        input_2_dims[0],
        "op_matmul_cpu dim error. input_1 ([1 (prepended), %ld]) and input_2 "
        "([%ld, "
        "%ld]) cannot conduct matrix-matrix multiply.",
        input_1_dims[0],
        input_2_dims[0],
        input_2_dims[1]);
  } else if (input_1_ndim == 2 && input_2_ndim == 1) {
    // If the first argument is 2-dimensional and the second argument is
    // 1-dimensional, the matrix-vector product is returned.

    FORCE_EQ(
        input_1_dims[1],
        input_2_dims[0],
        "op_matmul_cpu dim error. input_1 ([%ld, %ld]) and input_2 ([%ld]) "
        "cannot conduct matrix-vector multiply.",
        input_1_dims[0],
        input_1_dims[1],
        input_2_dims[0]);
  } else {
    // If both arguments are at least 1-dimensional and at least one argument
    // is N-dimensional (where N > 2), then a batched matrix multiply is
    // returned.
  }
}
