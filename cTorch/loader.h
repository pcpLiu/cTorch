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

#ifndef C_TORCH_LOADER_H
#define C_TORCH_LOADER_H

#include "cTorch/graph.h"

/*
  Load a graph from a fbs path.
  If any error happens, program will directly exit.

  Note: caller needs to release this plan manually.
*/
CTHGraph *load_graph(char *);

#endif /* LOADER_H */
