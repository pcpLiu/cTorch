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

#ifndef CTH_OP_LIST_H
#define CTH_OP_LIST_H

#include "cTorch/operators/default/op_list.h"

#ifdef BACKEND_CPU_X86
#include "cTorch/operators/x86/op_list.h"
#endif

#ifdef BACKEND_CPU_ARM
#include "cTorch/operators/arm/op_list_arm.h"
#endif

#ifdef BACKEND_OPENBLAS
#include "cTorch/operators/openblas/op_list.h"
#endif

#ifdef BACKEND_MKL
#include "cTorch/operators/mkl/op_list_mkl.h"
#endif

#ifdef BACKEND_CUDA
#include "cTorch/operators/cuda/op_list_cuda.h"
#endif

#ifdef BACKEND_APPLE
#include "cTorch/operators/apple/op_list_apple.h"
#endif

#endif /* OP_LIST_H */
