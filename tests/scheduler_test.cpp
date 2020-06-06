#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchSchedulerTest, testCreate) {
  CTorchGraph *graph = create_dummy_graph();
  int num_nodes = 10;
  for (int i = 0; i < num_nodes; i++) {
    insert_list(CTorchNode)(graph->node_list, create_dummy_node());
  }

  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = 4;

  CTorchScheduler *scheduler = cth_new_scheduler(config, graph);
  EXPECT_EQ(scheduler->job_list->size, num_nodes);
}

TEST(cTorchSchedulerTest, testGetJobForNode) {
  List(CTorchQueueJob) *job_list = new_list(CTorchQueueJob)();
  CTorchNode *node_1 = (CTorchNode *)MALLOC(sizeof(CTorchNode));
  CTorchQueueJob *job = (CTorchQueueJob *)MALLOC(sizeof(CTorchQueueJob));
  job->node = node_1;
  insert_list(CTorchQueueJob)(job_list, job);

  CTorchQueueJob *ret_job = cth_get_job_for_node(node_1, job_list, false);
  EXPECT_EQ(ret_job, job);

  CTorchNode *node_2 = (CTorchNode *)MALLOC(sizeof(CTorchNode));
  ret_job = cth_get_job_for_node(node_2, job_list, false);
  EXPECT_EQ(ret_job, nullptr);

  EXPECT_EXIT(cth_get_job_for_node(node_2, job_list, true),
              ::testing::ExitedWithCode(1),
              "Cannot find node in given job list");
}

CTorchQueueJob *dumm_job(CTorchNode *node, CTH_JOB_STATUS status) {
  CTorchQueueJob *job = (CTorchQueueJob *)MALLOC(sizeof(CTorchQueueJob));
  job->node = node;
  job->status = status;
  return job;
}

TEST(cTorchSchedulerTest, testSearchReadyJob) {
  /**
   *  node_1[x] --> node_2[x] --> node_4 --> node_5
   *    |                           ^
   *    |------> node_3[x] ---------|
   */

  CTorchNode *node_1 = create_dummy_node();
  CTorchNode *node_2 = create_dummy_node();
  CTorchNode *node_3 = create_dummy_node();
  CTorchNode *node_4 = create_dummy_node();
  CTorchNode *node_5 = create_dummy_node();

  insert_list(CTorchNode)(node_2->inbound_nodes, node_1);
  insert_list(CTorchNode)(node_3->inbound_nodes, node_1);
  insert_list(CTorchNode)(node_4->inbound_nodes, node_2);
  insert_list(CTorchNode)(node_4->inbound_nodes, node_3);
  insert_list(CTorchNode)(node_5->inbound_nodes, node_4);

  CTorchQueueJob *job_1 = dumm_job(node_1, CTH_JOB_STATUS_DONE);
  CTorchQueueJob *job_2 = dumm_job(node_2, CTH_JOB_STATUS_DONE);
  CTorchQueueJob *job_3 = dumm_job(node_3, CTH_JOB_STATUS_DONE);
  CTorchQueueJob *job_4 = dumm_job(node_4, CTH_JOB_STATUS_WAIT);
  CTorchQueueJob *job_5 = dumm_job(node_5, CTH_JOB_STATUS_WAIT);

  List(CTorchQueueJob) *queue_job_list = new_list(CTorchQueueJob)();
  List(CTorchQueueJob) *done_job_list = new_list(CTorchQueueJob)();
  List(CTorchQueueJob) *ready_job_list = new_list(CTorchQueueJob)();

  insert_list(CTorchQueueJob)(done_job_list, job_1);
  insert_list(CTorchQueueJob)(done_job_list, job_2);
  insert_list(CTorchQueueJob)(done_job_list, job_3);

  insert_list(CTorchQueueJob)(queue_job_list, job_4);
  insert_list(CTorchQueueJob)(queue_job_list, job_5);

  cth_search_ready_jobs(queue_job_list, done_job_list, ready_job_list);

  EXPECT_TRUE(list_contains_data(CTorchQueueJob)(ready_job_list, job_4) !=
              nullptr);
  EXPECT_TRUE(list_contains_data(CTorchQueueJob)(ready_job_list, job_5) ==
              nullptr);
}