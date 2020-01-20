#ifndef CTH_ENGINE_H
#define CTH_ENGINE_H

#include "plan.h"

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
void execute_node(CTorchNode *);

#endif /* ENGINE_H */
