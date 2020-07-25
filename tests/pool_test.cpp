#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#include <unistd.h>

TEST(cTorchPoolTest, testCreate) {
  CTHConfig *config = (CTHConfig *)MALLOC(sizeof(CTHConfig));
  config->num_workers = 4;

  cth_array_index_t num_nodes = 1000;
  CTHGraph *graph = create_dummy_graph(num_nodes);
  for (cth_array_index_t i = 0; i < num_nodes; i++) {
    cth_array_set(CTHNode)(graph->node_list, i, create_dummy_node(i, 0, 0));
  }

  CTHScheduler *scheduler = cth_new_scheduler(config, graph);
  // create threads in pool
  CTHWorkerPool *pool = cth_new_pool(scheduler, config);

  // Add job manually
  for (cth_array_index_t i = 0; i < num_nodes; i++) {
    CTHQueueJob *job = cth_array_at(CTHQueueJob)(scheduler->job_list, i);
    write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTHQueueJob *));
  }

  // check finish
  for (cth_array_index_t i = 0; i < num_nodes; i++) {
    CTHQueueJob *job = cth_array_at(CTHQueueJob)(scheduler->job_list, i);
    read(scheduler->ret_queue->pipe_fd[0], &job, sizeof(CTHQueueJob *));
    EXPECT_EQ(CTH_JOB_STATUS_DONE, job->status);
  }

  // kill threads
  for (cth_thread_n_t i = 0; i < config->num_workers; i++) {
    CTHQueueJob *job = (CTHQueueJob *)MALLOC(sizeof(job));
    job->worker_kill = true;
    write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTHQueueJob *));
  }
}

TEST(cTorchPoolTest, testKill) {
  CTHConfig *config = (CTHConfig *)MALLOC(sizeof(CTHConfig));
  config->num_workers = 4;

  cth_array_index_t num_nodes = 1000;
  CTHGraph *graph = create_dummy_graph(num_nodes);
  for (cth_array_index_t i = 0; i < num_nodes; i++) {
    cth_array_set(CTHNode)(graph->node_list, i, create_dummy_node(i, 0, 0));
  }

  CTHScheduler *scheduler = cth_new_scheduler(config, graph);
  CTHWorkerPool *pool = cth_new_pool(scheduler, config);

  // Add job manually
  for (cth_array_index_t i = 0; i < num_nodes; i++) {
    CTHQueueJob *job = cth_array_at(CTHQueueJob)(scheduler->job_list, i);
    write(scheduler->exe_queue->pipe_fd[1], &job, sizeof(CTHQueueJob *));
  }

  // check finish
  for (cth_array_index_t i = 0; i < num_nodes; i++) {
    CTHQueueJob *job = cth_array_at(CTHQueueJob)(scheduler->job_list, i);
    read(scheduler->ret_queue->pipe_fd[0], &job, sizeof(CTHQueueJob *));
    EXPECT_EQ(CTH_JOB_STATUS_DONE, job->status);
  }

  cth_close_pool(scheduler, pool);
}
