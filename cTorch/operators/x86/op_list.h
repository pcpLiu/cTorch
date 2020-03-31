#ifndef OP_LIST_X86_H
#define OP_LIST_X86_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"

/*
  Declare x86 op functions
*/
#define X86_OP_FUNC_NAME(op) op_##op##_x86
#define X86_OP_FUNC_DECLARE(op) void X86_OP_FUNC_NAME(op)(CTorchOperator *);
#define DECLARE_X86_ALL_OP_FUNCS FOREACH_OP_ID(X86_OP_FUNC_DECLARE)
DECLARE_X86_ALL_OP_FUNCS

#define X86_OP_FUNC_BIND(op) X86_OP_FUNC_NAME(op),
#define X86_ALL_OP_FUNCS FOREACH_OP_ID(X86_OP_FUNC_BIND)

/*
  Array of function pointers corresponding to each enabled operator
*/
extern void (*fps_op_x86[ENABLED_OP_NUM])(CTorchOperator *);

#endif /* OP_LIST_X86_H */
