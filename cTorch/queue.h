#ifndef CTH_QUEUE_H
#define CTH_QUEUE_H

#include "cTorch/operator.h"

#include <pthread.h>

/**
 * A queue used to communciate between threads. It's a wrap of POSIX PIPE. It
 * comes with two mutex to guard write & read.
 */
typedef struct CTorchQueue {
  int pipe_fd[2];              /* PIPE */
  pthread_mutex_t read_mutex;  /* Read mutex */
  pthread_mutex_t write_mutex; /* Write mutex */
  CTH_QUEUE_STATUS status;     /* Queue status */
} CTorchQueue;

/**
 * Create a new op queue
 */
CTorchQueue *cth_new_queue();

/**
 * Close both write & read ends, and set the status to inactive. It will destroy
 * two mutex
 */
void cth_close_queue(CTorchQueue *queue);

/**
 * Close queue and free the struct.
 */
void cth_close_and_free_queue(CTorchQueue *queue);

/**
 * Message to be passed through Queue
 */
typedef struct CTorchQueueMessage {
  CTorchOperator *op;    /* Operator */
  CTH_JOB_STATUS status; /* Status of this op execution */
  bool worker_kill; /* When this field is true, worker stop execution loop */
} CTorchQueueMessage;

// List utils for CTorchTensor
def_list_item(CTorchQueueMessage);
def_list(CTorchQueueMessage);
declare_new_list_item_func(CTorchQueueMessage);
declare_new_list_func(CTorchQueueMessage);
declare_insert_list_func(CTorchQueueMessage);
declare_list_contains_data_func(CTorchQueueMessage);
declare_list_contains_item_func(CTorchQueueMessage);
declare_list_at_func(CTorchQueueMessage);
declare_list_pop_func(CTorchQueueMessage);
declare_free_list_func(CTorchQueueMessage);

#endif /* QUEUE_H */
