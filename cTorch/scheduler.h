// Copyright 2021 Zhonghao Liu
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
typedef struct CTHScheduler {
  CTHQueue *
      exe_queue; /* Queue for nodes to be executed. Scheduler will put nodes in
                      this queue and workers will fetch nodes from this queue */
  CTHQueue
      *ret_queue; /* Queue for executed nodes. Workers will put nodes in this
                      queue and scheduler will fetch nodes from this queue  */
  CTHArray(CTHQueueJob) *
      job_list;                  /* List of jobs executed by this scheduler */
  cth_bit_array_t *queue_status; /* Queue status of all jobs */
  cth_bit_array_t *done_status;  /* Done status of all jobs */
  cth_bit_array_t *ready_status; /* Ready status of all jobs */
} CTHScheduler;

/**
 * Create a new scheduler based on config and graph. Function will parse graph's
 * nodes and fill into job_list.
 *
 * Arguments
 *    - config: execution config
 *    - graph: a computation graph
 */
CTHScheduler *cth_new_scheduler(CTHConfig *config, CTHGraph *graph);

/**
 * Turn on scheduler to put job into pipe til all jobs are done
 */
void cth_start_scheduler(CTHScheduler *scheduler);

/**
 * Search ready jobs and insert into ready_jobs
 *
 * Arguments:
 *    - scheduler: scheduler
 *    - ready_jobs: list to insert results
 */
void cth_search_ready_jobs(
    CTHScheduler *scheduler, CTHList(CTHQueueJob) * ready_jobs);

#endif /* SCHEDULER_H */
