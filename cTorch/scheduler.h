#ifndef CTH_SCHEDULER_H
#define CTH_SCHEDULER_H

#include "cTorch/bit_array.h"
#include "cTorch/config.h"
#include "cTorch/graph.h"
#include "cTorch/node.h"
#include "cTorch/queue.h"

/**
 * Scheduler is used to arrange a graph's execution order
 */
typedef struct CTorchScheduler {
  CTorchQueue *
      exe_queue; /* Queue for nodes to be executed. Scheduler will put nodes in
                      this queue and workers will fetch nodes from this queue */
  CTorchQueue
      *ret_queue; /* Queue for executed nodes. Workers will put nodes in this
                      queue and scheduler will fetch nodes from this queue  */
  Array(CTorchQueueJob) *
      job_list;              /* List of jobs executed by this scheduler */
  bit_array_t *queue_status; /* Queue status of all jobs */
  bit_array_t *done_status;  /* Done status of all jobs */
  bit_array_t *ready_status; /* Ready status of all jobs */
} CTorchScheduler;

/**
 * Create a new scheduler based on config and graph. Function will parse graph's
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

/**
 * Search ready jobs and insert into ready_jobs
 *
 * Arguments:
 *    - scheduler: scheduler
 *    - ready_jobs: list to insert results
 */
void cth_search_ready_jobs(
    CTorchScheduler *scheduler, List(CTorchQueueJob) * ready_jobs);

#endif /* SCHEDULER_H */
