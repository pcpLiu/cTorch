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
