#include "cTorch/queue.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <fcntl.h>
#include <unistd.h>

impl_new_list_item_func(CTorchQueueJob);
impl_new_list_func(CTorchQueueJob);
impl_insert_list_func(CTorchQueueJob);
impl_list_contains_data_func(CTorchQueueJob);
impl_list_contains_item_func(CTorchQueueJob);
impl_list_at_func(CTorchQueueJob);
impl_list_pop_func(CTorchQueueJob);
impl_free_list_func(CTorchQueueJob);
impl_list_del_func(CTorchQueueJob);

impl_new_array_func(CTorchQueueJob);
impl_array_at_func(CTorchQueueJob);
impl_array_set_func(CTorchQueueJob);

void set_non_block(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

CTorchQueue *cth_new_queue() {
  CTorchQueue *queue = MALLOC(sizeof(CTorchQueue));
  pthread_mutex_init(&queue->read_mutex, NULL);
  pthread_mutex_init(&queue->write_mutex, NULL);

  if (pipe(queue->pipe_fd) == -1) {
    perror("Error: ");
    FAIL_EXIT(
        CTH_LOG_ERR,
        "Failed to create a CTorchQueue. See above error message.");
  }

  // // set as non-blocking
  // for (int i = 0; i < 2; i++) {
  //   int flags = fcntl(queue->pipe_fd[i], F_GETFL, 0);
  //   fcntl(queue->pipe_fd[i], F_SETFL, flags | O_NONBLOCK);
  // }

  return queue;
}