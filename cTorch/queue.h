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

#ifndef CTH_QUEUE_H
#define CTH_QUEUE_H

#include "cTorch/node.h"

#include <pthread.h>

/**
 * A queue used to communciate between threads. It's a wrap of POSIX PIPE. It
 * comes with two mutex to guard write & read.
 */
typedef struct CTHQueue {
  int pipe_fd[2];              /* PIPE */
  pthread_mutex_t read_mutex;  /* Read mutex */
  pthread_mutex_t write_mutex; /* Write mutex */
  CTH_QUEUE_STATUS status;     /* Queue status */
} CTHQueue;

/**
 * Create a new op queue
 */
CTHQueue *cth_new_queue();

/**
 * Close both write & read ends, and set the status to inactive. It will destroy
 * two mutex
 */
void cth_close_queue(CTHQueue *queue);

/**
 * Close queue and free the struct.
 */
void cth_close_and_free_queue(CTHQueue *queue);

/**
 * Message to be passed through Queue
 */
typedef struct CTHQueueJob {
  CTHNode *node;         /* Node to be executed */
  CTH_JOB_STATUS status; /* Status of this node execution */
  bool worker_kill; /* When this field is true, worker stop execution loop */
} CTHQueueJob;

// List utils for CTHTensor
cth_def_list_item(CTHQueueJob);
def_list(CTHQueueJob);
cth_declare_new_list_item_func(CTHQueueJob);
cth_declare_new_list_func(CTHQueueJob);
cth_declare_insert_list_func(CTHQueueJob);
cth_declare_list_contains_data_func(CTHQueueJob);
cth_declare_list_contains_item_func(CTHQueueJob);
cth_declare_list_at_func(CTHQueueJob);
cth_declare_list_pop_func(CTHQueueJob);
cth_declare_free_list_func(CTHQueueJob);
cth_declare_list_del_func(CTHQueueJob);

cth_def_array(CTHQueueJob);
cth_declare_new_array_func(CTHQueueJob);
cth_declare_array_at_func(CTHQueueJob);
cth_declare_array_set_func(CTHQueueJob);

#endif /* QUEUE_H */
