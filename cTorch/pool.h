#ifndef CTH_POOL_H
#define CTH_POOL_H

#include <pthread.h>
#include <stdlib.h>

/* Number of thread unit type */
typedef uint8_t thread_n_t;

/*
  Worker thread in the pool.
*/
typedef struct CTorchThread {
  pthread_t tid; /**< thread id */
} CTorchThread;

/*
  Thread pool.
*/
typedef struct CTorchThreadPool {
  thread_n_t min_threads; /* minimum number of worker threads */
  thread_n_t max_threads; /* maximum number of worker threads */
} CTorchThreadPool;

/*
  Galobal thread pool.
*/
extern CTorchThreadPool CTH_THR_POOL;

#endif /* POOL_H */
