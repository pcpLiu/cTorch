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

#ifndef OP_LIST_APPLE_H
#define OP_LIST_APPLE_H

#include "cTorch/consts.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"

/**
 * Declare cpu op functions. These three macros will finally expand to a list of
 * functions like:
 *    void op_XXXXXX_apple(CTHOperator *);
 */
#define APPLE_OP_FUNC_NAME(op) op_##op##_apple
#define APPLE_OP_FUNC_DECLARE(op) void APPLE_OP_FUNC_NAME(op)(CTHOperator *);
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
extern void (*fps_op_apple[ENABLED_OP_NUM])(CTHOperator *);

#endif /* OP_LIST_APPLE_H */
