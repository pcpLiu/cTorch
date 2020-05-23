#ifndef CTH_SCHEDULER_H
#define CTH_SCHEDULER_H

#include "cTorch/config.h"
#include "cTorch/graph.h"
#include "cTorch/node.h"
#include "cTorch/queue.h"

/**
 * Scheduler is used to arrange a graph's execution order
 */
typedef struct CTorchScheduler {
  CTorchQueue
      *ready_queue; /* Queue for ops to be executed. Scheduler will put op in
                       this queue and workers will fetch op from this queue */
  CTorchQueue
      *done_queue; /* Queue for executed ops. Workers will put op in this queue
                      and scheduelr will fetch op from this queue  */
} CTorchScheduler;

/**
 * Create a new scheduler
 *
 * Arguments
 *    - config: execution config
 */
CTorchScheduler *cth_new_scheduler(CTorchConfig *config);

/**
 * Start schedule the execution of a graph
 */
void cth_scheduler_start(CTorchScheduler *scheduler, CTorchGraph *graph);

#endif /* SCHEDULER_H */
