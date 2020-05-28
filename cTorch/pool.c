#include "cTorch/pool.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <unistd.h>

void cth_worker_consume(CTorchQueueJob *msg) {
  // TODO: IMPL
  msg->status = CTH_JOB_STATUS_DONE;
}

void *cth_worker(void *scheduler_v) {
  /**
   * Loop til a killer switch message fetched:
   *    - Fetch messsage from ready_queue and execute it
   *    - Update message's job status and put it to done_queue
   */

  CTorchScheduler *scheduler = (CTorchScheduler *)scheduler_v;
  CTorchQueue *ready_queue = scheduler->ready_queue;
  CTorchQueue *done_queue = scheduler->done_queue;

  CTorchQueueJob *msg;
  while (true) {
    // TODO: do we need lock?
    pthread_mutex_lock(&ready_queue->read_mutex);
    read(ready_queue->pipe_fd[0], &msg, sizeof(CTorchQueueJob *));
    pthread_mutex_unlock(&ready_queue->read_mutex);

    if (msg->worker_kill) {
      break;
    }

    cth_worker_consume(msg);

    // TODO: do we need lock?
    pthread_mutex_lock(&done_queue->write_mutex);
    write(done_queue->pipe_fd[1], &msg, sizeof(CTorchQueueJob *));
    pthread_mutex_unlock(&done_queue->write_mutex);
  }

  pthread_exit(NULL);
}

CTorchWorkerPool *
cth_new_pool(CTorchScheduler *scheduler, CTorchConfig *config) {
  FAIL_NULL_PTR(scheduler);
  FAIL_NULL_PTR(config);

  CTorchWorkerPool *pool = MALLOC(sizeof(CTorchWorkerPool));
  pool->num_workers = config->num_workers;
  pool->workers = MALLOC(pool->num_workers * sizeof(pthread_t));

  // Set threads detached
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (thread_n_t thread_i = 0; thread_i < pool->num_workers; thread_i++) {
    int ret = pthread_create(
        &pool->workers[thread_i], &attr, cth_worker, (void *)scheduler);

    if (ret) {
      FAIL_EXIT(
          CTH_LOG_ERR,
          "Failed to create thread from pthread_create(). Error code: %d",
          ret);
    }
  }

  return pool;
}

void cth_close_pool(CTorchScheduler *scheduler, CTorchWorkerPool *pool) {
  FAIL_NULL_PTR(scheduler);
  FAIL_NULL_PTR(pool);

  for (thread_n_t i = 0; i < pool->num_workers; i++) {
    CTorchQueueJob *job = MALLOC(sizeof(CTorchQueueJob));
    job->worker_kill = true;
    write(scheduler->ready_queue->pipe_fd[1], &job, sizeof(CTorchQueueJob *));
  }

  // wait till all killed
  int err;
  void *status;
  for (thread_n_t i = 0; i < pool->num_workers; i++) {
    err = pthread_join(*(pool->workers + i), &status);
    if (err) {
      FAIL_EXIT(
          CTH_LOG_ERR,
          "pthread_join failed; return code from pthread_join() is %d\n",
          err);
    }
  }
}