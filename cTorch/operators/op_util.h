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

#ifndef CTH_OP_UTIL_H
#define CTH_OP_UTIL_H

/**
 * @brief Get input value should be multiplied with kernel value at [kernel_i,
 * kernel_z, kernel_x] contributing value for output at [batch_i, kernel_i,
 * output_x]. To locate the proper input value, we need to consider padding,
 * stride and dilation.
 *
 * @param batch_i cth_tensor_dim_t batch index
 * @param output_x cth_tensor_dim_t output value index on x dim
 * @param input_z cth_tensor_dim_t input value index on z dim
 * @param kernel_i cth_tensor_dim_t ith kernel
 * @param kernel_x cth_tensor_dim_t kernel value index on x dim
 * @param padding_mode CTH_PADDING_MODE padding mode
 * @param padding_left cth_tensor_dim_t left padding size
 * @param padding_right cth_tensor_dim_t right padding size
 * @param kernel_size cth_tensor_dim_t kernel size
 * @param dilation cth_tensor_dim_t dilation value
 * @param stride cth_tensor_dim_t stride value
 * @param in_ptr input value pointer
 * @param input_x_size cth_tensor_dim_t input tensor x dim size
 * @param input_val_var variable to hold the input value, can be any data type
 */
#define _cth_get_input_value_1d_conv(                                          \
    bacth_i,                                                                   \
    output_x,                                                                  \
    input_z,                                                                   \
    kernel_i,                                                                  \
    kernel_x,                                                                  \
    padding_mode,                                                              \
    padding_left,                                                              \
    padding_right,                                                             \
    kernel_size,                                                               \
    dilation,                                                                  \
    stride,                                                                    \
    in_ptr,                                                                    \
    input_x_size,                                                              \
    input_val_var)                                                             \
  do {                                                                         \
    cth_tensor_dim_t input_loc_offset =                                        \
        output_x * stride + kernel_x * dilation;                               \
    if (input_loc_offset < padding_left ||                                     \
        input_loc_offset >= padding_left + input_x_size) {                     \
      if (CTH_PADDING_MODE_ZEROS == padding_mode) {                            \
        input_val_var = 0;                                                     \
      } else if (CTH_PADDING_MODE_REFLECT == padding_mode) {                   \
        if (input_loc_offset < padding_left) {                                 \
          FORCE_TRUE(                                                          \
              padding_left < input_x_size,                                     \
              "In reflect padding, left padding size %d should be less than "  \
              "input dimension %d",                                            \
              padding_left,                                                    \
              input_x_zie);                                                    \
          input_val_var = in_ptr[input_loc_offset + 1];                        \
        } else {                                                               \
          FORCE_TRUE(                                                          \
              padding_right < input_x_size,                                    \
              "In reflect padding, right padding size %d should be less than " \
              "input dimension %d",                                            \
              padding_right,                                                   \
              input_x_zie);                                                    \
          input_val_ar = in_ptr                                                \
              [input_x_size - 1 -                                              \
               (input_loc_offset + 1 - padding_left - input_x_size)];          \
        }                                                                      \
      } else if (CTH_PADDING_MODE_REPLICATE == padding_mode) {                 \
        input_val_var = ;                                                      \
      } else if (CTH_PADDING_MODE_CIRCULAR == padding_mode) {                  \
        input_val_var = ;                                                      \
      } else {                                                                 \
        FAIL_EXIT(CTH_LOG_ERR, "Unsupported padding mode %d", padding_mode);   \
      }                                                                        \
    } else {                                                                   \
      input_val_var = in_ptr[input_loc_offset - padding_left];                 \
    }                                                                          \
  } while (0)

#endif /* OP_UTIL_H */
