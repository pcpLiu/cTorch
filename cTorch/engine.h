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
typedef struct CTHEngine {
  CTHScheduler *scheduler; /* Execution scheduler */
  CTHWorkerPool *pool;     /* Thread pool to execute operators */
} CTHEngine;

/**
 * Create an engine to execute graph. In this function, what cTorch will do:
 *    - Initialize a thread pool to execute ops
 *    - Initialzie a scheduler
 *    - Create pipes used to communicate between pool and main process
 *
 * Arguments:
 *    - config: execution config
 */
CTHEngine *cth_new_engine(CTHConfig *config);

void cth_execute_node(CTHNode *node, CTH_BACKEND backend);

#endif /* ENGINE_H */
