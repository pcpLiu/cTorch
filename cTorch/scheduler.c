#include "cTorch/scheduler.h"
#include "cTorch/mem_util.h"

#include <unistd.h>

/********************************************************************************
 * A simple quick sort impl
 */

static void
_swap_job(ListItem(CTorchQueueJob) * a, ListItem(CTorchQueueJob) * b) {
  CTorchQueueJob *t = a->data;
  a->data = b->data;
  b->data = t;
}

static ListItem(CTorchQueueJob) *
    _partition(ListItem(CTorchQueueJob) * l, ListItem(CTorchQueueJob) * h) {
  node_id_t x = h->data->node->node_id;
  ListItem(CTorchQueueJob) *i = l->prev_item;

  for (ListItem(CTorchQueueJob) *j = l; j != h; j = j->next_item) {
    if (j->data->node->node_id <= x) {
      i = (i == NULL) ? l : i->next_item;
      _swap_job(i, j);
    }
  }
  i = (i == NULL) ? l : i->next_item;
  _swap_job(i, h);
  return i;
}

static void _quick_sort_job_lis(
    ListItem(CTorchQueueJob) * l, ListItem(CTorchQueueJob) * h) {
  if (h != NULL && l != h && l != h->next_item) {
    ListItem(CTorchQueueJob) *p = _partition(l, h);
    _quick_sort_job_lis(l, p->prev_item);
    _quick_sort_job_lis(p->next_item, h);
  }
}

/**
 * Sort jobs list by node_id
 */
static void _sort_job_list(List(CTorchQueueJob) * jobs) {
  _quick_sort_job_lis(jobs->head, jobs->tail);
}

/*
 * End impl of quick sort
 ******************************************************************************/

CTorchScheduler *cth_new_scheduler(CTorchConfig *config, CTorchGraph *graph) {
  FAIL_NULL_PTR(config);
  FAIL_NULL_PTR(graph);

  CTorchScheduler *scheduler = MALLOC(sizeof(CTorchScheduler));
  scheduler->ret_queue = cth_new_queue();
  scheduler->exe_queue = cth_new_queue();
  scheduler->queue_status = cth_new_bit_array(graph->node_list->size);
  scheduler->done_status = cth_new_bit_array(graph->node_list->size);
  scheduler->ready_status = cth_new_bit_array(graph->node_list->size);

  /**
   * Fill job_list. Here we make sure the jobs in this list are sorted by
   * node_id.
   */
  scheduler->job_list = new_list(CTorchQueueJob)();
  for (list_index_t i = 0; i < graph->node_list->size; i++) {
    CTorchQueueJob *job = MALLOC(sizeof(CTorchQueueJob));
    job->node = list_at(CTorchNode)(graph->node_list, i);
    job->status = CTH_JOB_STATUS_WAIT;
    job->worker_kill = false;
    insert_list(CTorchQueueJob)(scheduler->job_list, job);
    cth_set_bit(scheduler->queue_status, i);
  }
  _sort_job_list(scheduler->job_list);

  return scheduler;
}

void cth_search_ready_jobs(
    CTorchScheduler *scheduler, List(CTorchQueueJob) * ready_jobs) {
  bit_array_t *queue_status = scheduler->queue_status;
  bit_array_t *done_status = scheduler->done_status;
  bit_array_t *ready_status = scheduler->ready_status;

  CTorchNode *queue_node = NULL;
  CTorchQueueJob *queue_job = NULL;
  for (bit_array_index_t queue_job_id = 0; queue_job_id < queue_status->size;
       queue_job_id++) {
    /* Job is not in queue */
    if (!cth_is_bit_set(queue_status, queue_job_id)) {
      continue;
    }

    /**
     * A queue node is ready:
     *    - All its inbounds nodes are ready, this node is ready;
     *    - Or, it has no inbounds nodes
     */

    /* This assumes job_list is sorted by node_id */
    queue_job = list_at(CTorchQueueJob)(scheduler->job_list, queue_job_id);
    queue_node = queue_job->node;

    bool job_ready = true;
    for (list_index_t i = 0; i < queue_node->inbound_nodes->size; i++) {
      // node_id_t dependent_node_id =
      //     list_at(CTorchNode)(queue_node->inbound_nodes, i)->node_id;
      node_id_t dependent_node_id =
          array_at(CTorchNode)(queue_node->inbound_nodes, i)->node_id;

      /* Dependent is not done. Queue job is not ready */
      if (!cth_is_bit_set(done_status, dependent_node_id)) {
        job_ready = false;
        break;
      }
    }

    if (job_ready) {
      /* Mark status from queue to ready */
      cth_clear_bit(scheduler->queue_status, queue_job->node->node_id);
      cth_set_bit(scheduler->ready_status, queue_job->node->node_id);
      queue_job->status = CTH_JOB_STATUS_READY;
      insert_list(CTorchQueueJob)(ready_jobs, queue_job);
    }
  }
}

void cth_start_scheduler_v3(CTorchScheduler *scheduler) {
  FAIL_NULL_PTR(scheduler);

  List(CTorchQueueJob) *ready_jobs = new_list(CTorchQueueJob)();
  CTorchQueueJob *job = NULL;

  /**
   * Scheduler main loop:
   *    - Find ready nodes, write them into ready queue
   *    - Wait on nodes popping up from done queue
   *    - Terminate loop when all jobs done
   */
  while (!cth_are_all_bits_set(scheduler->done_status)) {
    cth_search_ready_jobs(scheduler, ready_jobs);
    while (ready_jobs->size > 0) {
      job = list_pop(CTorchQueueJob)(ready_jobs);
      write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTorchQueueJob *));
    }

    read(scheduler->ret_queue->pipe_fd[0], &job, sizeof(CTorchQueueJob *));
    /* Mark status from ready to done */
    cth_clear_bit(scheduler->ready_status, job->node->node_id);
    cth_set_bit(scheduler->done_status, job->node->node_id);
  }
}

void cth_start_scheduler(CTorchScheduler *scheduler) {
  // cth_start_scheduler_v1(scheduler);
  // cth_start_scheduler_v2(scheduler);
  cth_start_scheduler_v3(scheduler);
}
