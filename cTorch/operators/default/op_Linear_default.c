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
 * @brief Computation logic.
 *
 * @param op CTHOperator operator
 * @param data_type  C type
 *
 */
#define _cth_linear(op, data_type)                                             \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *weights = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);     \
    CTHTensor *bias = cth_array_at(CTHTensor)(op->in_bound_tensors, 2);        \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
                                                                               \
    data_type *in_ptr = (data_type *)input->values;                            \
    data_type *weights_ptr = (data_type *)weights->values;                     \
    data_type *bias_ptr = (data_type *)bias->values;                           \
    data_type *out_ptr = (data_type *)output->values;                          \
                                                                               \
    cth_tensor_dim_t out_feature_dim = weights->meta_info->dims[0];            \
    cth_tensor_dim_t in_feature_dim = weights->meta_info->dims[1];             \
    cth_tensor_dim_t n_dim = input->meta_info->n_dim;                          \
                                                                               \
    cth_tensor_dim_t *in_dims = input->meta_info->dims;                        \
    cth_tensor_dim_t *out_dims = output->meta_info->dims;                      \
                                                                               \
    FORCE_EQ(                                                                  \
        input->meta_info->n_dim,                                               \
        output->meta_info->n_dim,                                              \
        "_cth_linear force equal fails. input & output should have same "      \
        "n_dim. But input n_dim: %ld, output n_dim: %ld",                      \
        input->meta_info->n_dim,                                               \
        output->meta_info->n_dim);                                             \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < n_dim - 1; i++) {                         \
      FORCE_EQ(                                                                \
          in_dims[i],                                                          \
          out_dims[i],                                                         \
          "_cth_linear force equal fails. input dims[%ld]: %ld, output "       \
          "dims[%ld]: %ld.",                                                   \
          i,                                                                   \
          in_dims[i],                                                          \
          i,                                                                   \
          out_dims[i]);                                                        \
    }                                                                          \
                                                                               \
    FORCE_EQ(                                                                  \
        in_dims[n_dim - 1],                                                    \
        in_feature_dim,                                                        \
        "_cth_linear force equal fails. input dims[%ld]: %ld, in_feature_dim " \
        "of weights tensor:%ld",                                               \
        n_dim - 1,                                                             \
        in_dims[n_dim - 1],                                                    \
        in_feature_dim);                                                       \
                                                                               \
    FORCE_EQ(                                                                  \
        out_dims[n_dim - 1],                                                   \
        out_feature_dim,                                                       \
        "_cth_linear force equal fails. output dims[%ld]: %ld, "               \
        "out_feature_dim "                                                     \
        "of weights tensor:%ld",                                               \
        n_dim - 1,                                                             \
        out_dims[n_dim - 1],                                                   \
        out_feature_dim);                                                      \
                                                                               \
    cth_tensor_dim_t iterations = 1;                                           \
    for (cth_tensor_dim_t i = 0; i < n_dim - 1; i++) {                         \
      iterations *= in_dims[i];                                                \
    }                                                                          \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < iterations; i++) {                        \
      cth_tensor_dim_t out_offset = i * out_feature_dim;                       \
      cth_tensor_dim_t in_offset = i * in_feature_dim;                         \
                                                                               \
      for (cth_tensor_dim_t x_out = 0; x_out < out_feature_dim; x_out++) {     \
        cth_tensor_dim_t weights_offset = x_out * in_feature_dim;              \
                                                                               \
        double out_val = 0.0;                                                  \
        for (cth_tensor_dim_t x_in = 0; x_in < in_feature_dim; x_in++) {       \
          out_val += (double)weights_ptr[weights_offset + x_in] *              \
                     (double)in_ptr[in_offset + x_in];                         \
        }                                                                      \
                                                                               \
        out_ptr[out_offset + x_out] = (data_type)(out_val + bias_ptr[x_out]);  \
      }                                                                        \
    }                                                                          \
                                                                               \
  } while (0);

/**
 * @brief Applies a linear transformation to the incoming data.
 *
 * @param op Operator
 *
 * @note Bias tensor always exists. During conversion, if bias is disabled, make
 * the tensor all veluas 0.
 *
 * @par Inputs & outputs:
 *   - # of input: 3
 *    - 0: input tensor, [batch (b), *, in_feature_dim (x_in)]
 *    - 1: weights, [out_features_dim, in_features_dim]
 *    - 2: bias, [out_features_dim]
 *   - # of output: 1
 *    - 0: output tensor, [batch (b), *, out_feature_dim (x_out)]
 */
void op_Linear_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 3, 1);
  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);

  _cpu_generic_compute(op, _cth_linear, input->meta_info->data_type);
}
