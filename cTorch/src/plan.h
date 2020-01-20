#ifndef CTH_PLAN_H
#define CTH_PLAN_H

#include "graph.h"
#include "node.h"

/*
  CTorchExecuteStep.
  This struct represent a basic executable unit for engine.
  It contails a list of nodes that could be executed simultaneously.
*/
typedef struct {
  // List of executable nodes in this step
  List(CTorchNode) * executable_nodes;

  // Step index
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
  // A list of executable steps, in order.
  List(CTorchExecuteStep) * steps;

  // Input tensors.
  // Caller should fill values of these tensors
  List(CTorchTensor) * inputs;

  // Output tensors.
  // Graph results are stored in this tensors
  List(CTorchTensor) * outputs;
} CTorchExecutePlan;

/*
  Build a plan from a graph based on depdendencies.

  Note: caller needs to release this plan manually.
*/
CTorchExecutePlan *build_plan(CTorchGraph *);

#endif /* PLAN_H */
