#ifndef CTH_PLAN_H
#define CTH_PLAN_H

#include "node.h"

/*
  CTorchExecuteStep.
  This struct represent a basic executable unit for engine.
  It contails a list of nodes that could be executed simultaneously.
*/
typedef struct {
  List(CTorchNode) * executable_nodes;
  uint16_t step_index;
} CTorchExecuteStep;

// step list macro
def_list_item(CTorchExecuteStep);
def_list(CTorchExecuteStep);
declare_new_list_item_func(CTorchExecuteStep);
declare_new_list_func(CTorchExecuteStep);
declare_insert_list_func(CTorchExecuteStep);
declare_list_contains_data_func(CTorchExecuteStep);
declare_list_contains_item_func(CTorchExecuteStep);

/*
  CTorchExecutePlan.
  A execution plan of a computaional graph.
  It will be consumed by engine.
*/
typedef struct {
  List(CTorchExecuteStep) * steps;
} CTorchExecutePlan;

#endif /* PLAN_H */
