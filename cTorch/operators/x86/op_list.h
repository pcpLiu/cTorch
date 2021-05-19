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

#ifndef OP_LIST_X86_H
#define OP_LIST_X86_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"

/*
  Declare x86 op functions
*/
#define X86_OP_FUNC_NAME(op) op_##op##_x86
#define X86_OP_FUNC_DECLARE(op) void X86_OP_FUNC_NAME(op)(CTHOperator *);
#define DECLARE_X86_ALL_OP_FUNCS FOREACH_OP_ID(X86_OP_FUNC_DECLARE)
DECLARE_X86_ALL_OP_FUNCS

#define X86_OP_FUNC_BIND(op) X86_OP_FUNC_NAME(op),
#define X86_ALL_OP_FUNCS FOREACH_OP_ID(X86_OP_FUNC_BIND)

/*
  Array of function pointers corresponding to each enabled operator
*/
extern void (*fps_op_x86[ENABLED_OP_NUM])(CTHOperator *);

#endif /* OP_LIST_X86_H */
