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

#include "cTorch/queue.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <fcntl.h>
#include <unistd.h>

cth_impl_new_list_item_func(CTHQueueJob);
cth_impl_new_list_func(CTHQueueJob);
cth_impl_insert_list_func(CTHQueueJob);
cth_impl_list_contains_data_func(CTHQueueJob);
cth_impl_list_contains_item_func(CTHQueueJob);
cth_impl_list_at_func(CTHQueueJob);
cth_impl_list_pop_func(CTHQueueJob);
cth_impl_free_list_func(CTHQueueJob);
cth_impl_list_del_func(CTHQueueJob);

cth_impl_new_array_func(CTHQueueJob);
cth_impl_array_at_func(CTHQueueJob);
cth_impl_array_set_func(CTHQueueJob);

void set_non_block(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

CTHQueue *cth_new_queue() {
  CTHQueue *queue = MALLOC(sizeof(CTHQueue));
  pthread_mutex_init(&queue->read_mutex, NULL);
  pthread_mutex_init(&queue->write_mutex, NULL);

  if (pipe(queue->pipe_fd) == -1) {
    perror("Error: ");
    FAIL_EXIT(
        CTH_LOG_ERR, "Failed to create a CTHQueue. See above error message.");
  }

  // // set as non-blocking
  // for (int i = 0; i < 2; i++) {
  //   int flags = fcntl(queue->pipe_fd[i], F_GETFL, 0);
  //   fcntl(queue->pipe_fd[i], F_SETFL, flags | O_NONBLOCK);
  // }

  return queue;
}
