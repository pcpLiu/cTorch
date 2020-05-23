#ifndef CTH_ENGINE_H
#define CTH_ENGINE_H

#include "cTorch/config.h"
#include "cTorch/consts.h"
#include "cTorch/node.h"
#include "cTorch/pool.h"
#include "cTorch/queue.h"
#include "cTorch/scheduler.h"

/**
 * Execution engine
 */
typedef struct CTorchEngine {
  CTorchScheduler *scheduler; /* Execution scheduler */
  CTorchWorkerPool *pool;     /* Thread pool to execute operators */
} CTorchEngine;

/**
 * Create an engine to execute graph. In this function, what cTorch will do:
 *    - Initialize a thread pool to execute ops
 *    - Initialzie a scheduler
 *    - Create pipes used to communicate between pool and main process
 *
 * Arguments:
 *    - config: execution config
 */
CTorchEngine *cth_new_engine(CTorchConfig *config);

void cth_execute_node(CTorchNode *node, CTH_BACKEND backend);

#endif /* ENGINE_H */
