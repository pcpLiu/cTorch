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

#ifndef CTH_CONSTS_H
#define CTH_CONSTS_H

#include "cTorch/ops_enabled.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/*
  Backends
*/
typedef enum CTH_BACKEND {
  CTH_BACKEND_DEFAULT,
  CTH_BACKEND_MKL,
  CTH_BACKEND_OPENBLAS,
  CTH_BACKEND_APPLE,
  CTH_BACKEND_CPU_X86,
  CTH_BACKEND_CPU_ARM,
  CTH_BACKEND_CUDA,
} CTH_BACKEND;

/**
 * Graph node type:
 *    - CTH_NODE_TYPE_DATA: tensor
 *    - CTH_NODE_TYPE_OPERATOR: operator
 */
typedef enum CTH_NODE_TYPE {
  CTH_NODE_TYPE_DATA,
  CTH_NODE_TYPE_OPERATOR,
} CTH_NODE_TYPE;

/**
 * Tensor value types
 * ref: https://pytorch.org/docs/stable/tensors.html#torch-tensor
 */
typedef enum CTH_TENSOR_DATA_TYPE {
  CTH_TENSOR_DATA_TYPE_FLOAT_16 = 0,
  CTH_TENSOR_DATA_TYPE_FLOAT_32,
  CTH_TENSOR_DATA_TYPE_FLOAT_64,
  CTH_TENSOR_DATA_TYPE_INT_8,
  CTH_TENSOR_DATA_TYPE_INT_16,
  CTH_TENSOR_DATA_TYPE_INT_32,
  CTH_TENSOR_DATA_TYPE_INT_64,
  CTH_TENSOR_DATA_TYPE_UINT_8,
  CTH_TENSOR_DATA_TYPE_BOOL,
} CTH_TENSOR_DATA_TYPE;

/*
  Tensor type:
    - CTH_TENSOR_TYPE_NORMAL: non-parameter tensors
    - CTH_TENSOR_TYPE_PARAM: parameter tensors
*/
typedef enum CTH_TENSOR_TYPE {
  CTH_TENSOR_TYPE_NORMAL,
  CTH_TENSOR_TYPE_PARAM,
} CTH_TENSOR_TYPE;

/**
 * @brief Tensor physical device
 *
 * Types:
 *    - CTH_TENSOR_DEVICE_NORMAL: normal tensor lives in main mem
 *    - CTH_TENSOR_DEVICE_CUDA: tensor lives in CUDA mem
 */
typedef enum CTH_TENSOR_DEVICE {
  CTH_TENSOR_DEVICE_NORMAL,
  CTH_TENSOR_DEVICE_CUDA,
} CTH_TENSOR_DEVICE;

/*
  Job execution status:
    - CTH_JOB_STATUS_WAIT: job is waiting for depedency
    - CTH_JOB_STATUS_READY: job is to be executed
    - CTH_JOB_STATUS_DONE: job is done
*/
typedef enum CTH_JOB_STATUS {
  CTH_JOB_STATUS_WAIT,
  CTH_JOB_STATUS_READY,
  CTH_JOB_STATUS_DONE,
} CTH_JOB_STATUS;

/**
 * Queue status:
 *    - CTH_QUEUE_STATUS_ALIVE: queue is active
 *    - CTH_QUEUE_STATUS_INACTIVE: queue is inactive. Should not been used in
 *                                 this status
 */
typedef enum CTH_QUEUE_STATUS {
  CTH_QUEUE_STATUS_ALIVE,
  CTH_QUEUE_STATUS_INACTIVE,
} CTH_QUEUE_STATUS;

/**
 * @brief Padding modes
 *
 */
typedef enum CTH_PADDING_MODE {
  CTH_PADDING_MODE_ZEROS,
  CTH_PADDING_MODE_REFLECT,
  CTH_PADDING_MODE_REPLICATE,
  CTH_PADDING_MODE_CIRCULAR,
} CTH_PADDING_MODE;

/**
 * @brief Type to denote bool type
 */
typedef bool cth_bool_t;

/**
 * Type to denote thread num
 */
typedef uint16_t cth_thread_n_t;

/**
 * Type to denote tensor dimension & size
 */
typedef int64_t cth_tensor_dim_t;

/**
 * @brief Type to denote generic float parameter of an operator.
 */
typedef float cth_float_param_t;

/**
 * @brief Type to denote channel param. Used in conv, pooling etc.
 */
typedef cth_tensor_dim_t cth_channel_t;

/**
 * @brief Type to denote kernel size param. Used in conv, pooling etc.
 */
typedef cth_tensor_dim_t cth_kernel_t;

/**
 * @brief Type to denote paddin param. Used in conv, pooling etc.
 */
typedef cth_tensor_dim_t cth_pad_t;

/**
 * @brief Type to denote stride param. Used in conv, pooling etc.
 */
typedef cth_tensor_dim_t cth_stride_t;

/**
 * @brief Type to denote dilation param. Used in conv, pooling etc.
 */
typedef cth_tensor_dim_t cth_dilation_t;

/**
 * @brief Type to denote groups param. Used in conv, pooling etc.
 */
typedef cth_tensor_dim_t cth_groups_t;

/**
 * @brief Index type of result tensro in reduce index op
 */
#define cth_tensor_reduce_index_t int64_t

/*
  To define operator enums
*/
#define _ENUMFY_OP(x) CTH_OP_ID_##x,

/*
  To declare array of op names
*/
#define _STRINGTIFY_OP(x) #x,

/**
 * Universal struct deep free function name in cToRCH
 */
#define struct_deep_free(data_type) cth_free_deep_##data_type

/**
 * Pi value. 10 digits
 */
#define CTH_PI 3.1415926535

/**
 * @brief Cast ptr to another type
 */
#define CTH_CAST_PTR(ptr, type) ((type *)ptr)

/**
 * @brief Deference a pointer or a pointer expression
 *
 */
#define PTR_VAL(ptr_expression) (*(ptr_expression))

/*
  String array of operator names.

  It is one-to-one indexed for enum CTH_OPERATOR_ID
*/
extern char *CTH_OPERATOR_NAMES[ENABLED_OP_NUM];

/**
 * @brief Operator ID enum
 * @note ending ';' is required
 */
typedef enum CTH_OP_ID { FOREACH_OP_ID(_ENUMFY_OP) } CTH_OP_ID;

#endif /* CONSTS_H */
