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

#ifndef CTH_PARAMS_H
#define CTH_PARAMS_H

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"

/**
 * Enum used to indicate param type
 */
typedef enum CTH_PARAM_TYPE {
  /* multiplier */
  CTH_PARAM_TYPE_MULTIPLIER,
  CTH_PARAM_TYPE_MIN,
  CTH_PARAM_TYPE_MAX,
  CTH_PARAM_TYPE_P,
  CTH_PARAM_TYPE_DIM,
  CTH_PARAM_TYPE_IN_CHANNELS,
  CTH_PARAM_TYPE_OUT_CHANNELS,
  CTH_PARAM_TYPE_KERNEL_SIZE,
  CTH_PARAM_TYPE_STRIDE,
  CTH_PARAM_TYPE_PADDING_D2,
  CTH_PARAM_TYPE_PADDING_D4,
  CTH_PARAM_TYPE_PADDING_D6,
  CTH_PARAM_TYPE_DILATION,
  CTH_PARAM_TYPE_KERNEL_SIZE_D2,
  CTH_PARAM_TYPE_STRIDE_D2,
  CTH_PARAM_TYPE_DILATION_D2,
  CTH_PARAM_TYPE_PADDING_MODE,
  CTH_PARAM_TYPE_GROUPS,
  /* Padding value. Store as float. */
  CTH_PARAM_TYPE_PADDING_VALUE_FLOAT,
} CTH_PARAM_TYPE;

/**
 * Param data union
 */
typedef union CTHParamData {
  cth_float_param_t *multiplier;
  cth_float_param_t *min;
  cth_float_param_t *max;
  cth_float_param_t *p;
  cth_channel_t *dim;
  cth_channel_t *in_channels;
  cth_channel_t *out_channels;
  cth_stride_t *stride;
  cth_kernel_t *kernel_size;
  cth_pad_t *padding;
  cth_dilation_t *dilation;
  cth_groups_t *groups;
  CTH_PADDING_MODE *padding_mode;
  cth_float_param_t *padding_value_float;
} CTHParamData;

typedef struct CTHParam {
  CTH_PARAM_TYPE type;
  CTHParamData data;
} CTHParam;

// Array macros
cth_def_array(CTHParam);
cth_declare_new_array_func(CTHParam);
cth_declare_array_at_func(CTHParam);
cth_declare_array_set_func(CTHParam);
cth_declare_free_array_deep_func(CTHParam);

/**
 * Deep free a CTHParam
 *
 * Note:
 *    If pointer is NULL, error raised and exit.
 */
void struct_deep_free(CTHParam)(CTHParam *param);

/**
 * Copy fields from `from_param` to `to_param`
 */
void cth_copy_param(CTHParam *from_param, CTHParam *to_param);

/**
 * @brief Extract param value with given types. Will raise error if param not
 * exist.
 *
 * @note The `param_var` should be a pointer. `param_var` will just be assigned
 * the meem address of existing parameter variable. It does not copy the value
 * of parameters.
 */
#define EXTRACT_PARAM_VALUE(op, param_type, param_data_field, param_var)       \
  do {                                                                         \
    CTHParam *param = cth_get_param_by_type(op, param_type, true);             \
    param_var = param->data.param_data_field;                                  \
  } while (0)

/**
 * @brief Extract param value with given types. Assign param as NULL if not
 * found.
 *
 * @note The param_var should be a pointer.
 */
#define EXTRACT_PARAM_VALUE_OR_NULL(                                           \
    op, param_type, param_data_field, param_var_ptr)                           \
  do {                                                                         \
    CTHParam *param = cth_get_param_by_type(op, param_type, true);             \
    if (param != NULL) {                                                       \
      param_var_ptr = param->data.param_data_field;                            \
    } else {                                                                   \
      param_var_ptr = NULL;                                                    \
    }                                                                          \
  } while (0)

#endif /* PARAMS_H */
