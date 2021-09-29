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
  CTH_PARAM_TYPE_MULTIPLIER,     /* multiplier, float */
  CTH_PARAM_TYPE_MIN,            /* min, float */
  CTH_PARAM_TYPE_MAX,            /* max, float */
  CTH_PARAM_TYPE_P,              /* p, float */
  CTH_PARAM_TYPE_DIM,            /* Axes dimension, cth_tensor_dim_t */
  CTH_PARAM_TYPE_IN_CHANNELS,    /* # of input channels, cth_tensor_dim_t */
  CTH_PARAM_TYPE_OUT_CHANNELS,   /* # of out channels, cth_tensor_dim_t */
  CTH_PARAM_TYPE_GROUPS,         /* # of groups, cth_tensor_dim_t */
  CTH_PARAM_TYPE_KERNEL_SIZE,    /* 1-D kernel size , cth_tensor_dim_t */
  CTH_PARAM_TYPE_STRIDE,         /* 1-D stride size , cth_tensor_dim_t */
  CTH_PARAM_TYPE_DILATION,       /* 1-D dilation size , cth_tensor_dim_t */
  CTH_PARAM_TYPE_KERNEL_SIZE_D2, /* 2-D kernel size , CTHDim2 */
  CTH_PARAM_TYPE_STRIDE_D2,      /* 2-D stride size , CTHDim2 */
  CTH_PARAM_TYPE_DILATION_D2,    /* 2-D dilation size , CTHDim2 */
  CTH_PARAM_TYPE_PADDING_D2,     /* 2-D padding size , CTHDim2 */
  CTH_PARAM_TYPE_PADDING_D4,     /* 4-D padding size , CTHDim4 */
  CTH_PARAM_TYPE_PADDING_D6,     /* 6-D padding size , CTHDim6 */
  CTH_PARAM_TYPE_PADDING_MODE,   /* padding mode, CTH_PADDING_MODE*/
  CTH_PARAM_TYPE_PADDING_VALUE_FLOAT,  /* Padding value. float */
  CTH_PARAM_TYPE_ALPHA_FLOAT,          /* alpha, float */
  CTH_PARAM_TYPE_LAMBD_FLOAT,          /* lambda, float */
  CTH_PARAM_TYPE_NEGATIVE_SLOPE_FLOAT, /* negative slope, float */
} CTH_PARAM_TYPE;

/**
 * @brief 2-D dimentions struct
 *
 */
typedef struct CTHDim2 {
  cth_tensor_dim_t d_0;
  cth_tensor_dim_t d_1;
} CTHDim2;

/**
 * @brief 4-D dimentions struct
 *
 */
typedef struct CTHDim4 {
  cth_tensor_dim_t d_0;
  cth_tensor_dim_t d_1;
  cth_tensor_dim_t d_2;
  cth_tensor_dim_t d_3;
} CTHDim4;

/**
 * @brief 6-D dimentions struct
 *
 */
typedef struct CTHDim6 {
  cth_tensor_dim_t d_0;
  cth_tensor_dim_t d_1;
  cth_tensor_dim_t d_2;
  cth_tensor_dim_t d_3;
  cth_tensor_dim_t d_4;
  cth_tensor_dim_t d_5;
} CTHDim6;

/**
 * Param data union
 */
typedef union CTHParamData {
  cth_float_param_t *float_val;
  cth_tensor_dim_t *dim_val;
  cth_bool_t *bool_val;
  CTHDim2 *dim_2_val;
  CTHDim4 *dim_4_val;
  CTHDim6 *dim_6_val;
  CTH_PADDING_MODE *padding_mode_val;
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

#endif /* PARAMS_H */
