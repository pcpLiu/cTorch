#ifndef OP_LIST_DEFAULT_H
#define OP_LIST_DEFAULT_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Declare cpu op functions. These three macros will finally expand to a list of
 * functions like:
 *    void op_XXXXXX_cpu(CTHOperator *);
 */
#define CPU_OP_FUNC_NAME(op) op_##op##_cpu
#define CPU_OP_FUNC_DECLARE(op) void CPU_OP_FUNC_NAME(op)(CTHOperator *);
#define DECLARE_CPU_ALL_OP_FUNCS FOREACH_OP_ID(CPU_OP_FUNC_DECLARE)
DECLARE_CPU_ALL_OP_FUNCS

/**
 *
 */
#define CPU_OP_FUNC_BIND(op) CPU_OP_FUNC_NAME(op),
#define CPU_ALL_OP_FUNCS FOREACH_OP_ID(CPU_OP_FUNC_BIND)

/**
 * Array of function pointers corresponding to each enabled operator
 */
extern void (*fps_op_default[ENABLED_OP_NUM])(CTHOperator *);

#endif /* OP_LIST_DEFAULT_H */
