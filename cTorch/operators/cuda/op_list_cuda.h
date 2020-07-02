#ifndef OP_LIST_CUDA_H
#define OP_LIST_CUDA_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Declare cpu op functions. These three macros will finally expand to a list of
 * functions like:
 *    void op_XXXXXX_cpu(CTorchOperator *);
 */
#define CUDA_OP_FUNC_NAME(op) op_##op##_cuda
#define CUDA_OP_FUNC_DECLARE(op) void CUDA_OP_FUNC_NAME(op)(CTorchOperator *);
#define DECLARE_CUDA_ALL_OP_FUNCS FOREACH_OP_ID(CUDA_OP_FUNC_DECLARE)
DECLARE_CUDA_ALL_OP_FUNCS

/**
 *
 */
#define CUDA_OP_FUNC_BIND(op) CUDA_OP_FUNC_NAME(op),
#define CUDA_ALL_OP_FUNCS FOREACH_OP_ID(CUDA_OP_FUNC_BIND)

/**
 * Array of function pointers corresponding to each enabled operator
 */
extern void (*fps_op_cuda[ENABLED_OP_NUM])(CTorchOperator *);

#endif /* OP_LIST_CUDA_H */
