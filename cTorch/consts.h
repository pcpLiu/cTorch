#ifndef CTH_CONSTS_H
#define CTH_CONSTS_H

#include "cTorch/ops_enabled.h"

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
 * Type to denote thread num
 */
typedef uint16_t thread_n_t;

/*
  To define operator enums
*/
#define ENUMFY_OP(x) CTH_OP_ID_##x,

/*
  To declare array of op names
*/
#define STRINGTIFY_OP(x) #x,

/**
 * Universal struct deep free function name in cToRCH
 */
#define struct_deep_free(data_type) free_deep_##data_type

/**
 * Type to denote tensor dimension
 */
#define tensor_dim_t uint32_t

/**
 * Type to denote tensor size
 */
#define tensor_size_t uint32_t

/**
 * Pi value. 10 digits
 */
#define CTH_PI 3.1415926535

/*
  String array of operator names.

  It is one-to-one indexed for enum CTH_OPERATOR_ID
*/
extern char *CTH_OPERATOR_NAMES[ENABLED_OP_NUM];

/*
  Operator ID enum
*/
typedef enum CTH_OP_ID { FOREACH_OP_ID(ENUMFY_OP) } CTH_OP_ID;

#endif /* CONSTS_H */
