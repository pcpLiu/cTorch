/**
 * Copyright 2021 Zhonghao Liu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cTorch/pool.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <unistd.h>

void cth_worker_consume(CTHQueueJob *msg) {
  // TODO: IMPL
  msg->status = CTH_JOB_STATUS_DONE;
}

void *cth_worker(void *scheduler_v) {
  /**
   * Loop til a killer switch message fetched:
   *    - Fetch messsage from exe_queue and execute it
   *    - Update message's job status and put it to ret_queue
   */
  FAIL_NULL_PTR(scheduler_v);

  CTHScheduler *scheduler = (CTHScheduler *)scheduler_v;
  CTHQueue *exe_queue = scheduler->exe_queue;
  CTHQueue *ret_queue = scheduler->ret_queue;

  CTHQueueJob *msg;
  while (true) {
    // TODO: do we need lock?
    pthread_mutex_lock(&exe_queue->read_mutex);
    read(exe_queue->pipe_fd[0], &msg, sizeof(CTHQueueJob *));
    pthread_mutex_unlock(&exe_queue->read_mutex);

    if (msg->worker_kill) {
      break;
    }

    cth_worker_consume(msg);

    // TODO: do we need lock?
    pthread_mutex_lock(&ret_queue->write_mutex);
    write(ret_queue->pipe_fd[1], &msg, sizeof(CTHQueueJob *));
    pthread_mutex_unlock(&ret_queue->write_mutex);
  }

  pthread_exit(NULL);
}

CTHWorkerPool *cth_new_pool(CTHScheduler *scheduler, CTHConfig *config) {
  FAIL_NULL_PTR(scheduler);
  FAIL_NULL_PTR(config);

  CTHWorkerPool *pool = MALLOC(sizeof(CTHWorkerPool));
  pool->num_workers = config->num_workers;
  pool->workers = MALLOC(pool->num_workers * sizeof(pthread_t));

  // Set threads detached
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (cth_thread_n_t thread_i = 0; thread_i < pool->num_workers; thread_i++) {
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

void cth_close_pool(CTHScheduler *scheduler, CTHWorkerPool *pool) {
  FAIL_NULL_PTR(scheduler);
  FAIL_NULL_PTR(pool);
  for (cth_thread_n_t i = 0; i < pool->num_workers; i++) {
    CTHQueueJob *job = MALLOC(sizeof(CTHQueueJob));
    job->worker_kill = true;
    write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTHQueueJob *));
  }

  // wait till all killed
  int err;
  void *status;
  for (cth_thread_n_t i = 0; i < pool->num_workers; i++) {
    err = pthread_join(*(pool->workers + i), &status);
    if (err) {
      FAIL_EXIT(
          CTH_LOG_ERR,
          "pthread_join failed; return code from pthread_join() is %d\n",
          err);
    }
  }
}
