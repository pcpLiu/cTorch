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

#ifndef CTH_GRAPH_H
#define CTH_GRAPH_H

#include "cTorch/consts.h"
#include "cTorch/node.h"
#include <stdint.h>

/**
 * This struct represents a computational graph.
 */
typedef struct {
  char *graph_name;              /* Graph name. Optional */
  CTHArray(CTHNode) * node_list; /* Nodes containing in this graph */
} CTHGraph;

#endif /* GRAPH_H */
