/**
 * Copyright 2021 Zhonghao Liu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cTorch/operators/default/op_list.h"

// void (*fps_op_x86[ENABLED_OP_NUM])(CTHOperator *) = {X86_ALL_OP_FUNCS};
void (*fps_op_default[ENABLED_OP_NUM])(CTHOperator *) = {
    op_abs_cpu, op_acos_cpu, op_add_cpu, op_addcdiv_cpu};
