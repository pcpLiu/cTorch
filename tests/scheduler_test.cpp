#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchSchedulerTest, testCreate) {
  CTorchGraph *graph = create_dummy_graph();
  int num_nodes = 10;
  for (int i = 0; i < num_nodes; i++) {
    insert_list(CTorchNode)(graph->node_list, create_dummy_node(i, 0, 0));
  }

  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = 4;

  CTorchScheduler *scheduler = cth_new_scheduler(config, graph);
  EXPECT_EQ(scheduler->job_list->size, num_nodes);
  EXPECT_TRUE(cth_are_all_bits_clear(scheduler->done_status));
  EXPECT_TRUE(cth_are_all_bits_clear(scheduler->ready_status));
  EXPECT_TRUE(cth_are_all_bits_set(scheduler->queue_status));
}

TEST(cTorchSchedulerTest, testSearchReadyJob) {
  /**
   *  node_1[x] --> node_2[x] --> node_4 --> node_5
   *    |                           ^
   *    |------> node_3[x] ---------|
   */

  CTorchNode *node_1 = create_dummy_node(0, 0, 2);
  CTorchNode *node_2 = create_dummy_node(1, 1, 1);
  CTorchNode *node_3 = create_dummy_node(2, 1, 1);
  CTorchNode *node_4 = create_dummy_node(3, 2, 1);
  CTorchNode *node_5 = create_dummy_node(4, 1, 0);

  array_set(CTorchNode)(node_2->inbound_nodes, 0, node_1);
  array_set(CTorchNode)(node_3->inbound_nodes, 0, node_1);
  array_set(CTorchNode)(node_4->inbound_nodes, 0, node_2);
  array_set(CTorchNode)(node_4->inbound_nodes, 1, node_3);
  array_set(CTorchNode)(node_5->inbound_nodes, 0, node_4);

  CTorchGraph *graph = create_dummy_graph();
  insert_list(CTorchNode)(graph->node_list, node_1);
  insert_list(CTorchNode)(graph->node_list, node_2);
  insert_list(CTorchNode)(graph->node_list, node_3);
  insert_list(CTorchNode)(graph->node_list, node_4);
  insert_list(CTorchNode)(graph->node_list, node_5);

  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = 4;
  CTorchScheduler *scheduler = cth_new_scheduler(config, graph);

  // Manually update status
  cth_set_bit(scheduler->done_status, 0);
  cth_clear_bit(scheduler->ready_status, 0);
  cth_clear_bit(scheduler->queue_status, 0);

  cth_set_bit(scheduler->done_status, 1);
  cth_clear_bit(scheduler->ready_status, 1);
  cth_clear_bit(scheduler->queue_status, 1);

  cth_set_bit(scheduler->done_status, 2);
  cth_clear_bit(scheduler->ready_status, 2);
  cth_clear_bit(scheduler->queue_status, 2);

  List(CTorchQueueJob) *ready_jobs = new_list(CTorchQueueJob)();
  cth_search_ready_jobs(scheduler, ready_jobs);

  // NODE 4 ready
  EXPECT_EQ(cth_is_bit_set(scheduler->ready_status, 3), true);
  EXPECT_EQ(cth_is_bit_set(scheduler->queue_status, 3), false);
  EXPECT_EQ(cth_is_bit_set(scheduler->done_status, 3), false);
  EXPECT_EQ(ready_jobs->size, 1);
  EXPECT_EQ(ready_jobs->head->data->node, node_4);

  // node 5 queue
  EXPECT_EQ(cth_is_bit_set(scheduler->ready_status, 4), false);
  EXPECT_EQ(cth_is_bit_set(scheduler->queue_status, 4), true);
  EXPECT_EQ(cth_is_bit_set(scheduler->done_status, 3), false);
}

TEST(cTorchSchedulerTest, testStartScheduler) {
  /**
   *  node_1 --> node_2 --> node_4 --> node_5
   *    |                      ^
   *    |------> node_3 -------|
   */

  CTorchNode *node_1 = create_dummy_node(0, 0, 2);
  CTorchNode *node_2 = create_dummy_node(1, 1, 1);
  CTorchNode *node_3 = create_dummy_node(2, 1, 1);
  CTorchNode *node_4 = create_dummy_node(3, 2, 1);
  CTorchNode *node_5 = create_dummy_node(4, 1, 0);

  array_set(CTorchNode)(node_2->inbound_nodes, 0, node_1);
  array_set(CTorchNode)(node_3->inbound_nodes, 0, node_1);
  array_set(CTorchNode)(node_4->inbound_nodes, 0, node_2);
  array_set(CTorchNode)(node_4->inbound_nodes, 1, node_3);
  array_set(CTorchNode)(node_5->inbound_nodes, 0, node_4);

  CTorchGraph *graph = create_dummy_graph();
  insert_list(CTorchNode)(graph->node_list, node_1);
  insert_list(CTorchNode)(graph->node_list, node_2);
  insert_list(CTorchNode)(graph->node_list, node_3);
  insert_list(CTorchNode)(graph->node_list, node_4);
  insert_list(CTorchNode)(graph->node_list, node_5);

  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = CPU_CORES;
  CTorchScheduler *scheduler = cth_new_scheduler(config, graph);
  CTorchWorkerPool *pool = cth_new_pool(scheduler, config);
  cth_start_scheduler(scheduler);
  cth_close_pool(scheduler, pool);

  // all bits are good
  EXPECT_EQ(cth_are_all_bits_clear(scheduler->queue_status), true);
  EXPECT_EQ(cth_are_all_bits_clear(scheduler->ready_status), true);
  EXPECT_EQ(cth_are_all_bits_set(scheduler->done_status), true);

  for (list_index_t i = 0; i < scheduler->job_list->size; i++) {
    CTorchQueueJob *job = list_at(CTorchQueueJob)(scheduler->job_list, i);
    EXPECT_EQ(job->status, CTH_JOB_STATUS_DONE);
  }
}

TEST(cTorchSchedulerTest, testManyTasks) {
  /**
   *  N nodes --> node_final
   */
  int N_DEPENDENTS = 100;

  CTorchGraph *graph = create_dummy_graph();

  CTorchNode *node_final = create_dummy_node(N_DEPENDENTS, N_DEPENDENTS, 0);
  insert_list(CTorchNode)(graph->node_list, node_final);

  for (int i = 0; i < N_DEPENDENTS; i++) {
    CTorchNode *node = create_dummy_node(i, 0, 1);
    array_set(CTorchNode)(node_final->inbound_nodes, i, node);
    insert_list(CTorchNode)(graph->node_list, node);
  }

  CTorchConfig *config = (CTorchConfig *)MALLOC(sizeof(CTorchConfig));
  config->num_workers = CPU_CORES;
  CTorchScheduler *scheduler = cth_new_scheduler(config, graph);
  CTorchWorkerPool *pool = cth_new_pool(scheduler, config);
  cth_start_scheduler(scheduler);
  cth_close_pool(scheduler, pool);
}