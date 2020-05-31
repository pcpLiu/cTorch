#include "cTorch/scheduler.h"
#include "cTorch/mem_util.h"

#include <unistd.h>

CTorchScheduler *cth_new_scheduler(CTorchConfig *config, CTorchGraph *graph) {
  FAIL_NULL_PTR(config);
  FAIL_NULL_PTR(graph);

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

void cth_start_scheduler(CTorchScheduler *scheduler) {
  FAIL_NULL_PTR(scheduler);

  /**
   * Scheduler loop:
   *    - Find ready jobs, put them into ready queue
   *    - Wait on jobs popping up in done queue
   *    - Terminate loop when all jobs done
   */

  List(CTorchQueueJob) *ready_job_list = new_list(CTorchQueueJob)();
  List(CTorchQueueJob) *done_job_list = new_list(CTorchQueueJob)();
  List(CTorchQueueJob) *queue_job_list = new_list(CTorchQueueJob)();
  for (list_index_t i = 0; i < scheduler->job_list->size; i++) {
    insert_list(CTorchQueueJob)(
        queue_job_list, list_at(CTorchQueueJob)(scheduler->job_list, i));
  }

  CTorchQueueJob *job = NULL;
  while (queue_job_list->size != 0) {
    cth_search_ready_jobs(queue_job_list, done_job_list, ready_job_list);
    for (list_index_t i = 0; i < ready_job_list->size; i++) {
      job = list_pop(CTorchQueueJob)(ready_job_list);
      job->status = CTH_JOB_STATUS_READY;
      write(scheduler->ready_queue->pipe_fd[1], &job, sizeof(CTorchQueueJob *));
    }

    read(scheduler->done_queue->pipe_fd[0], &job, sizeof(CTorchQueueJob *));
    insert_list(CTorchQueueJob)(done_job_list, job);
  }

  FREE(ready_job_list);
  FREE(done_job_list);
  FREE(queue_job_list);
}

void cth_search_ready_jobs(
    List(CTorchQueueJob) * queue_job_list,
    List(CTorchQueueJob) * done_job_list,
    List(CTorchQueueJob) * ready_job_list) {

  /**
   * Iterate queue_job_list from head to tail. During this process,
   * queue_job_list's size may chance while we are moving job from
   * queue_job_list to ready_job_list.
   */

  ListItem(CTorchQueueJob) *queue_job_item = queue_job_list->head;
  while (queue_job_item != NULL) {
    CTorchQueueJob *queue_job = queue_job_item->data;
    List(CTorchNode) *dependents = queue_job->node->inbound_nodes;

    bool job_ready = true;
    for (list_index_t j = 0; j < dependents->size; j++) {
      CTorchNode *dependent = list_at(CTorchNode)(dependents, j);
      if (cth_get_job_for_node(dependent, done_job_list, false) == NULL) {
        job_ready = false;
        break;
      }
    }

    if (job_ready) {
      insert_list(CTorchQueueJob)(ready_job_list, queue_job);
      list_del(CTorchQueueJob)(queue_job_list, queue_job);
    }

    queue_job_item = queue_job_item->next_item;
  }
}

CTorchQueueJob *cth_get_job_for_node(
    CTorchNode *node, List(CTorchQueueJob) * job_list, bool fail_not_found) {
  FAIL_NULL_PTR(node);
  FAIL_NULL_PTR(job_list);

  CTorchQueueJob *ret = NULL;
  for (list_index_t i = 0; i < job_list->size; i++) {
    CTorchQueueJob *job = list_at(CTorchQueueJob)(job_list, i);
    if (job->node == node) {
      ret = job;
      break;
    }
  }

  if (ret == NULL && fail_not_found) {
    FAIL_EXIT(CTH_LOG_ERR, "Cannot find node in given job list.");
  }

  return ret;
}