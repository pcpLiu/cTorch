#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

#include <unistd.h>

TEST(cTorchPoolTest, testWorkerFunc) {
  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = 4;
  CTorchScheduler *scheduler = cth_new_scheduler(config);
  CTorchWorkerPool *pool = cth_new_pool(scheduler, config);

  // Add job
  int n_job = 1000;
  for (int i = 0; i < n_job; i++) {
    CTorchQueueMessage *msg = (CTorchQueueMessage *)MALLOC(sizeof(msg));
    msg->status = CTH_JOB_STATUS_READY;
    msg->worker_kill = false;
    // std::cout << "[" << i << "]Create msg: " << msg
    //           << ", status: " << msg->status << std::endl;
    write(scheduler->ready_queue->pipe_fd[1], &msg,
          sizeof(CTorchQueueMessage *));
  }

  // check finish
  for (int i = 0; i < n_job; i++) {
    CTorchQueueMessage *msg;
    read(scheduler->done_queue->pipe_fd[0], &msg, sizeof(CTorchQueueMessage *));
    // std::cout << "[" << i << "]Get don msg: " << msg
    //           << ", status: " << msg->status << std::endl;
    EXPECT_EQ(CTH_JOB_STATUS_DONE, msg->status);
  }

  // kill threads
  for (thread_n_t i = 0; i < config->num_workers; i++) {
    CTorchQueueMessage *msg = (CTorchQueueMessage *)MALLOC(sizeof(msg));
    msg->worker_kill = true;
    write(scheduler->ready_queue->pipe_fd[1], &msg,
          sizeof(CTorchQueueMessage *));
  }
}