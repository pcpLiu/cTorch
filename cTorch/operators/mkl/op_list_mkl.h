#ifndef OP_LIST_MKL_H
#define OP_LIST_MKL_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Declare cpu op functions. These three macros will finally expand to a list of
 * functions like:
 *    void op_XXXXXX_mkl(CTHOperator *);
 */
#define MKL_OP_FUNC_NAME(op) op_##op##_mkl
#define MKL_OP_FUNC_DECLARE(op) void MKL_OP_FUNC_NAME(op)(CTHOperator *);
#define DECLARE_MKL_ALL_OP_FUNCS FOREACH_OP_ID(MKL_OP_FUNC_DECLARE)
DECLARE_MKL_ALL_OP_FUNCS

/**
 *
 */
#define MKL_OP_FUNC_BIND(op) MKL_OP_FUNC_NAME(op),
#define MKL_ALL_OP_FUNCS FOREACH_OP_ID(MKL_OP_FUNC_BIND)

/**
 * Array of function pointers corresponding to each enabled operator
 */
extern void (*fps_op_mkl[ENABLED_OP_NUM])(CTHOperator *);

#endif /* OP_LIST_MKL_H */
