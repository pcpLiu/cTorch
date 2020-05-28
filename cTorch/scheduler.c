#include "cTorch/scheduler.h"
#include "cTorch/mem_util.h"

CTorchScheduler *cth_new_scheduler(CTorchConfig *config, CTorchGraph *graph) {
  CTorchScheduler *scheduler = MALLOC(sizeof(CTorchScheduler));
  scheduler->done_queue = cth_new_queue();
  scheduler->ready_queue = cth_new_queue();

  scheduler->job_list = new_list(CTorchQueueJob)();
  for (int i = 0; i < graph->node_list->size; i++) {
    CTorchQueueJob *job = MALLOC(sizeof(CTorchQueueJob));
    job->node = list_at(CTorchNode)(graph->node_list, i);
    job->status = CTH_JOB_STATUS_WAIT;
    job->worker_kill = false;
    insert_list(CTorchQueueJob)(scheduler->job_list, job);
  }

  return scheduler;
}