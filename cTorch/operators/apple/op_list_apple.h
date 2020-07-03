#ifndef OP_LIST_APPLE_H
#define OP_LIST_APPLE_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Declare cpu op functions. These three macros will finally expand to a list of
 * functions like:
 *    void op_XXXXXX_apple(CTorchOperator *);
 */
#define APPLE_OP_FUNC_NAME(op) op_##op##_apple
#define APPLE_OP_FUNC_DECLARE(op) void APPLE_OP_FUNC_NAME(op)(CTorchOperator *);
#define DECLARE_APPLE_ALL_OP_FUNCS FOREACH_OP_ID(APPLE_OP_FUNC_DECLARE)
DECLARE_APPLE_ALL_OP_FUNCS

/**
 *
 */
#define APPLE_OP_FUNC_BIND(op) APPLE_OP_FUNC_NAME(op),
#define APPLE_ALL_OP_FUNCS FOREACH_OP_ID(APPLE_OP_FUNC_BIND)

/**
 * Array of function pointers corresponding to each enabled operator
 */
extern void (*fps_op_apple[ENABLED_OP_NUM])(CTorchOperator *);

#endif /* OP_LIST_APPLE_H */
