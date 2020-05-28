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
  List(CTorchQueueJob) * job_list; /* Full list of jobs (ops) to be executed */
} CTorchScheduler;

/**
 * Create a new scheduler based on cofig and graph. Function will parse graph's
 * nodes and fill into job_list.
 *
 * Arguments
 *    - config: execution config
 *    - graph: a computation graph
 */
CTorchScheduler *cth_new_scheduler(CTorchConfig *config, CTorchGraph *graph);

/**
 * Turn on scheduler to put job into pipe til all jobs are done
 */
void cth_start_scheduler(CTorchScheduler *scheduler);

#endif /* SCHEDULER_H */
