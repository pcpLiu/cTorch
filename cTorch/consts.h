#ifndef CTH_CONSTS_H
#define CTH_CONSTS_H

#include "cTorch/ops_enabled.h"

/*
  Backends
*/
typedef enum CTH_BACKEND {
  CTH_BACKEND_DEFAULT,
  CTH_BACKEND_MKL,
  CTH_BACKEND_OPENBLAS,
  CTH_BACKEND_ACCELERATE,
  CTH_BACKEND_CPU_X86,
  CTH_BACKEND_CPU_ARM,
  CTH_BACKEND_CUDA,
} CTH_BACKEND;

/*
  Graph node type:
    - CTH_NODE_TYPE_DATA: tensor
    - CTH_NODE_TYPE_OPERATOR: operator
*/
typedef enum CTH_NODE_TYPE {
  CTH_NODE_TYPE_DATA,
  CTH_NODE_TYPE_OPERATOR,
} CTH_NODE_TYPE;

/*
  Tensor value types
  ref: https://pytorch.org/docs/stable/tensors.html#torch-tensor
*/
typedef enum CTH_TENSOR_DATA_TYPE {
  // These types are compatable with pyTorch data dtypes.
  // Operators shouuld support these types.
  CTH_TENSOR_DATA_TYPE_FLOAT_16 = 0,
  CTH_TENSOR_DATA_TYPE_FLOAT_32,
  CTH_TENSOR_DATA_TYPE_FLOAT_64,
  CTH_TENSOR_DATA_TYPE_INT_8,
  CTH_TENSOR_DATA_TYPE_INT_16,
  CTH_TENSOR_DATA_TYPE_INT_32,
  CTH_TENSOR_DATA_TYPE_INT_64,
  CTH_TENSOR_DATA_TYPE_UINT_8,
  CTH_TENSOR_DATA_TYPE_BOOL,

  // These types are used by cTorch.
  // Operators doesn't need to support these types.
  CTH_TENSOR_DATA_TYPE_UINT_32,
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

/*
  Node execution status:
    - CTH_NODE_EXE_STATUS_CLEAN: wait to be executed
    - CTH_NODE_EXE_STATUS_DIRTY: executed
*/
typedef enum CTH_NODE_EXE_STATUS {
  CTH_NODE_EXE_STATUS_CLEAN,
  CTH_NODE_EXE_STATUS_DIRTY,
} CTH_NODE_EXE_STATUS;

/*
  To define operator enums
*/
#define ENUMFY_OP(x) CTH_OP_ID_##x,

/*
  To declare array of op names
*/
#define STRINGTIFY_OP(x) #x,

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
