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

#ifndef C_TORCH_LIBRARY_H
#define C_TORCH_LIBRARY_H

/* Export internal symbols in debug mode */
#ifdef CTH_TEST_DEBUG
#define CTH_EXPORT_INTERNAL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/bit_array.h"
#include "cTorch/consts.h"
#include "cTorch/debug_util.h"
#include "cTorch/engine.h"
#include "cTorch/generic_array.h"
#include "cTorch/graph.h"
#include "cTorch/list_d.h"
#include "cTorch/node.h"
#include "cTorch/operator.h"
#include "cTorch/operators/op_list.h"
#include "cTorch/pool.h"
#include "cTorch/queue.h"
#include "cTorch/scheduler.h"
#include "cTorch/sharder.h"
#include "cTorch/storage.h"

#ifdef __cplusplus
}
#endif

#endif // C_TORCH_LIBRARY_H
