#ifndef CTH_ENGINE_H
#define CTH_ENGINE_H

#include "graph.h"
#include "node.h"
#include <stdint.h>

typedef struct {
  List(CTorchNode) * executable_nodes;
  uint16_t n_nodes;
  uint16_t step_index;
} CTorchExecuteStep;

typedef struct {
  CTorchExecuteStep *steps;
  uint16_t n_steps;
} CTorchExecutePlan;

CTorchExecutePlan *ctorch_engine_build_plan(CTorchGraph *, CTorchExecutePlan *);
CTorchNode *torch_engine_execute_node(CTorchNode *);

#endif /* ENGINE_H */
