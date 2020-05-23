#ifndef CTH_POOL_H
#define CTH_POOL_H

#include "cTorch/config.h"
#include "cTorch/scheduler.h"

#include <pthread.h>

/**
 * Thread pool to execute operators
 */
typedef struct CTorchWorkerPool {
  pthread_t *workers;     /* thread array */
  thread_n_t num_workers; /* No. of workers */
} CTorchWorkerPool;

/**
 * Create a worker pool. Once created, all workers are waiting data from
 * scheduler til kill signals are sent.
 *
 * Arguments:
 *    - scheduelr: the scheduler used for this pool
 *    - config: execution config
 */
CTorchWorkerPool *
cth_new_pool(CTorchScheduler *scheduler, CTorchConfig *config);

/**
 * The worker function
 *
 * Arguments:
 *    - scheduler: a scheduler
 */
void *cth_worker(void *scheduler);

#endif /* POOL_H */
