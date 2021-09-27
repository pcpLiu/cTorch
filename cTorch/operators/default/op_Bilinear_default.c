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

/**
 * @brief Computation logic.
 *
 * @param op CTHOperator operator
 * @param data_type  C type
 *
 */
#define _cth_bilinear(op, data_type)                                           \
  do {                                                                         \
    CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);     \
    CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *weights = cth_array_at(CTHTensor)(op->in_bound_tensors, 2);     \
    CTHTensor *bias = cth_array_at(CTHTensor)(op->in_bound_tensors, 3);        \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
                                                                               \
    data_type *in_ptr_1 = (data_type *)input_1->values;                        \
    data_type *in_ptr_2 = (data_type *)input_2->values;                        \
    data_type *weights_ptr = (data_type *)weights->values;                     \
    data_type *bias_ptr = (data_type *)bias->values;                           \
    data_type *out_ptr = (data_type *)output->values;                          \
                                                                               \
    cth_tensor_dim_t out_feature_dim = weights->meta_info->dims[0];            \
    cth_tensor_dim_t in_feature_dim_1 = weights->meta_info->dims[1];           \
    cth_tensor_dim_t in_feature_dim_2 = weights->meta_info->dims[2];           \
    cth_tensor_dim_t n_dim = input_1->meta_info->n_dim;                        \
                                                                               \
    cth_tensor_dim_t *in_dims_1 = input_1->meta_info->dims;                    \
    cth_tensor_dim_t *in_dims_2 = input_2->meta_info->dims;                    \
    cth_tensor_dim_t *out_dims = output->meta_info->dims;                      \
                                                                               \
    FORCE_EQ(                                                                  \
        input_1->meta_info->n_dim,                                             \
        input_2->meta_info->n_dim,                                             \
        "_cth_bilinear dim error. input_1 & input_2 should have same "         \
        "n_dim. But input_1 n_dim: %ld, input_2 n_dim: %ld",                   \
        input_1->meta_info->n_dim,                                             \
        input_2->meta_info->n_dim);                                            \
                                                                               \
    FORCE_EQ(                                                                  \
        input_1->meta_info->n_dim,                                             \
        output->meta_info->n_dim,                                              \
        "_cth_bilinear dim error. input_1 & output should have same "          \
        "n_dim. But input_1 n_dim: %ld, output n_dim: %ld",                    \
        input_1->meta_info->n_dim,                                             \
        output->meta_info->n_dim);                                             \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < n_dim - 1; i++) {                         \
      FORCE_EQ(                                                                \
          in_dims_1[i],                                                        \
          in_dims_2[i],                                                        \
          "_cth_bilinear dim error. input_1 dims[%ld]: %ld, input_2 "          \
          "dims[%ld]: %ld.",                                                   \
          i,                                                                   \
          in_dims_1[i],                                                        \
          i,                                                                   \
          in_dims_2[i]);                                                       \
                                                                               \
      FORCE_EQ(                                                                \
          in_dims_1[i],                                                        \
          out_dims[i],                                                         \
          "_cth_bilinear dim error. input_1 dims[%ld]: %ld, output "           \
          "dims[%ld]: %ld.",                                                   \
          i,                                                                   \
          in_dims_1[i],                                                        \
          i,                                                                   \
          out_dims[i]);                                                        \
    }                                                                          \
                                                                               \
    FORCE_EQ(                                                                  \
        in_dims_1[n_dim - 1],                                                  \
        in_feature_dim_1,                                                      \
        "_cth_bilinear dim error. input_1 dims[%ld]: %ld, "                    \
        "in_feature_dim_1 "                                                    \
        "of weights tensor:%ld",                                               \
        n_dim - 1,                                                             \
        in_dims_1[n_dim - 1],                                                  \
        in_feature_dim_1);                                                     \
                                                                               \
    FORCE_EQ(                                                                  \
        in_dims_2[n_dim - 1],                                                  \
        in_feature_dim_2,                                                      \
        "_cth_bilinear dim error. input_2 dims[%ld]: %ld, "                    \
        "in_feature_dim_2 "                                                    \
        "of weights tensor:%ld",                                               \
        n_dim - 1,                                                             \
        in_dims_2[n_dim - 1],                                                  \
        in_feature_dim_2);                                                     \
                                                                               \
    FORCE_EQ(                                                                  \
        out_dims[n_dim - 1],                                                   \
        out_feature_dim,                                                       \
        "_cth_bilinear dim error. output dims[%ld]: %ld, "                     \
        "out_feature_dim "                                                     \
        "of weights tensor:%ld",                                               \
        n_dim - 1,                                                             \
        out_dims[n_dim - 1],                                                   \
        out_feature_dim);                                                      \
                                                                               \
    cth_tensor_dim_t iterations = 1;                                           \
    for (cth_tensor_dim_t i = 0; i < n_dim - 1; i++) {                         \
      iterations *= in_dims_1[i];                                              \
    }                                                                          \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < iterations; i++) {                        \
      cth_tensor_dim_t out_offset = i * out_feature_dim;                       \
      cth_tensor_dim_t in_offset_1 = i * in_feature_dim_1;                     \
      cth_tensor_dim_t in_offset_2 = i * in_feature_dim_2;                     \
                                                                               \
      for (cth_tensor_dim_t x_out = 0; x_out < out_feature_dim; x_out++) {     \
        cth_tensor_dim_t weights_offset =                                      \
            x_out * (in_feature_dim_1 * in_feature_dim_2);                     \
                                                                               \
        double out_val = 0.0;                                                  \
                                                                               \
        for (cth_tensor_dim_t x_in_2 = 0; x_in_2 < in_feature_dim_2;           \
             x_in_2++) {                                                       \
          double temp_matrix_val = 0.0;                                        \
                                                                               \
          for (cth_tensor_dim_t x_in_1 = 0; x_in_1 < in_feature_dim_1;         \
               x_in_1++) {                                                     \
            temp_matrix_val +=                                                 \
                (double)weights_ptr                                            \
                    [weights_offset + x_in_2 + (x_in_1 * in_feature_dim_2)] *  \
                (double)in_ptr_1[in_offset_1 + x_in_1];                        \
          }                                                                    \
                                                                               \
          out_val += temp_matrix_val * (double)in_ptr_2[in_offset_2 + x_in_2]; \
        }                                                                      \
                                                                               \
        out_ptr[out_offset + x_out] = (data_type)(out_val + bias_ptr[x_out]);  \
      }                                                                        \
    }                                                                          \
  } while (0);

/**
 * @brief Applies a bilinear transformation to the incoming data.
 *
 * @param op Operator
 *
 * @note Bias tensor always exists. During conversion, if bias is disabled, make
 * the tensor all veluas 0.
 *
 * @par Inputs & outputs:
 *   - # of input: 4
 *    - 0: input_1, [batch (b), *, in_feature_dim_1 (x_in)]
 *    - 1: input_2, [batch (b), *, in_feature_dim_2 (x_in)]
 *    - 2: weights, [out_features_dim, in_feature_dim_1, in_feature_dim_2]
 *    - 3: bias, [out_features_dim]
 *   - # of output: 1
 *    - 0: output tensor, [batch (b), *, out_feature_dim (x_out)]
 */
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

void op_Bilinear_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 4, 1);
  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);

  _cpu_generic_compute(op, _cth_bilinear, input->meta_info->data_type);
}
