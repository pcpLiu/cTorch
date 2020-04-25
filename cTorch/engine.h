#ifndef CTH_ENGINE_H
#define CTH_ENGINE_H

#include "cTorch/common.h"
#include "cTorch/consts.h"
#include "cTorch/operators/op_list.h"
#include "cTorch/plan.h"
#include "cTorch/pool.h"

/*
  Configs of execution engine.
*/
typedef struct CTorchEngineConfig {
  bool enable_sharding; /* If true, cTorch will execute operator in parallel
                           threads when possible. */
  thread_n_t num_max_threads; /* num of maximum threads. Only works when
                                 enable_sharding is True*/
} CTorchEngineConfig;

/* The global engine config. */
extern CTorchEngineConfig CTH_ENGINE_COFIG;

/*
  Execute a plan.
*/
void cth_execute_plan(CTorchExecutePlan *plan);

/*
  Execute a step.
*/
void cth_execute_step(CTorchExecuteStep *step);

/*
  Execute a node.
*/
void cth_execute_node(CTorchNode *, CTH_BACKEND);

#endif /* ENGINE_H */
