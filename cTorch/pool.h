#ifndef CTH_POOL_H
#define CTH_POOL_H

#include "cTorch/config.h"
#include "cTorch/scheduler.h"

#include <pthread.h>

/**
 * Thread pool to execute operators
 */
typedef struct CTHWorkerPool {
  pthread_t *workers;         /* thread array */
  cth_thread_n_t num_workers; /* No. of workers */
} CTHWorkerPool;

/**
 * Create a worker pool. Once created, all workers are waiting data from
 * scheduler til kill signals are sent.
 *
 * Arguments:
 *    - scheduelr: the scheduler used for this pool
 *    - config: execution config
 */
CTHWorkerPool *cth_new_pool(CTHScheduler *scheduler, CTHConfig *config);

/**
 * Close a work pool. Kill all working threads. This funcion use pthread_join,
 * so it will block till all jobs finished.
 *
 * Note: this func does not do any memory cleanning stuff.
 *
 * Arguments:
 *    - scheduler: attached scheduler
 *    - pool: pool to be closed
 */
void cth_close_pool(CTHScheduler *scheduler, CTHWorkerPool *pool);

/**
 * The worker function
 *
 * Arguments:
 *    - scheduler: a scheduler
 */
void *cth_worker(void *scheduler);

#endif /* POOL_H */
