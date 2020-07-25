#ifndef CTH_OP_LIST_OPENBLAS_H
#define CTH_OP_LIST_OPENBLAS_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"

#define OPENBLAS_OP_FUNC_NAME(op) op_##op_openblas

#define OPENBLAS_OP_FUNC_DECLARE(op)                                           \
  void OPENBLAS_OP_FUNC_NAME(op)(CTHOperator *)

#define DECLARE_OPENBLAS_ALL_OP_FUNCS                                          \
  FOREACH_OP_ID(OPENBLAS_OP_FUNC_DECLARE, SEMI_COL)

/*
  Array of function pointers corresponding to each enabled operator
*/
static void (*fps_op_openblas[ENABLED_OP_NUM])(CTHOperator *);

#endif /* OP_LIST_OPENBLAS_H */
