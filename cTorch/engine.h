#ifndef CTH_ENGINE_H
#define CTH_ENGINE_H

#include "cTorch/common.h"
#include "cTorch/consts.h"
#include "cTorch/operators/op_list.h"
#include "cTorch/plan.h"

/*
  Execute a plan.
*/
void execute_plan(CTorchExecutePlan *);

/*
  Execute a step.
*/
void execute_step(CTorchExecuteStep *);

/*
  Execute a node.
*/
void execute_node(CTorchNode *, CTH_BACKEND);

#endif /* ENGINE_H */
