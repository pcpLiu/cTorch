#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#include <unistd.h>

TEST(cTorchPoolTest, testCreate) {
  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = 4;

  int num_nodes = 1000;
  CTorchGraph *graph = create_dummy_graph(num_nodes);
  for (array_index_t i = 0; i < num_nodes; i++) {
    array_set(CTorchNode)(graph->node_list, i, create_dummy_node(i, 0, 0));
  }

  CTorchScheduler *scheduler = cth_new_scheduler(config, graph);
  CTorchWorkerPool *pool = cth_new_pool(scheduler, config);

  // Add job manually
  for (int i = 0; i < num_nodes; i++) {
    CTorchQueueJob *job = list_at(CTorchQueueJob)(scheduler->job_list, i);
    write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTorchQueueJob *));
  }

  // check finish
  for (int i = 0; i < num_nodes; i++) {
    CTorchQueueJob *job = list_at(CTorchQueueJob)(scheduler->job_list, i);
    read(scheduler->ret_queue->pipe_fd[0], &job, sizeof(CTorchQueueJob *));
    EXPECT_EQ(CTH_JOB_STATUS_DONE, job->status);
  }

  // kill threads
  for (thread_n_t i = 0; i < config->num_workers; i++) {
    CTorchQueueJob *job = (CTorchQueueJob *)MALLOC(sizeof(job));
    job->worker_kill = true;
    write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTorchQueueJob *));
  }
}

TEST(cTorchPoolTest, testKill) {
  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = 4;

  int num_nodes = 1000;
  CTorchGraph *graph = create_dummy_graph(num_nodes);
  for (array_index_t i = 0; i < num_nodes; i++) {
    array_set(CTorchNode)(graph->node_list, i, create_dummy_node(i, 0, 0));
  }

  CTorchScheduler *scheduler = cth_new_scheduler(config, graph);
  CTorchWorkerPool *pool = cth_new_pool(scheduler, config);

  // Add job manually
  for (int i = 0; i < num_nodes; i++) {
    CTorchQueueJob *job = list_at(CTorchQueueJob)(scheduler->job_list, i);
    write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTorchQueueJob *));
  }

  // check finish
  for (int i = 0; i < num_nodes; i++) {
    CTorchQueueJob *job = list_at(CTorchQueueJob)(scheduler->job_list, i);
    read(scheduler->ret_queue->pipe_fd[0], &job, sizeof(CTorchQueueJob *));
    EXPECT_EQ(CTH_JOB_STATUS_DONE, job->status);
  }

  cth_close_pool(scheduler, pool);
}