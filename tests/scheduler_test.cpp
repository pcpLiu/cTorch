#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchSchedulerTest, testCreate) {
  cth_array_index_t num_nodes = 10;
  CTHGraph *graph = create_dummy_graph(num_nodes);
  for (cth_array_index_t i = 0; i < num_nodes; i++) {
    cth_array_set(CTHNode)(graph->node_list, i, create_dummy_node(i, 0, 0));
  }

  CTHConfig *config = (CTHConfig *)MALLOC(sizeof(CTHConfig));
  config->num_workers = 4;

  CTHScheduler *scheduler = cth_new_scheduler(config, graph);
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

  CTHNode *node_1 = create_dummy_node(0, 0, 2);
  CTHNode *node_2 = create_dummy_node(1, 1, 1);
  CTHNode *node_3 = create_dummy_node(2, 1, 1);
  CTHNode *node_4 = create_dummy_node(3, 2, 1);
  CTHNode *node_5 = create_dummy_node(4, 1, 0);

  cth_array_set(CTHNode)(node_2->inbound_nodes, 0, node_1);
  cth_array_set(CTHNode)(node_3->inbound_nodes, 0, node_1);
  cth_array_set(CTHNode)(node_4->inbound_nodes, 0, node_2);
  cth_array_set(CTHNode)(node_4->inbound_nodes, 1, node_3);
  cth_array_set(CTHNode)(node_5->inbound_nodes, 0, node_4);

  CTHGraph *graph = create_dummy_graph(5);
  cth_array_set(CTHNode)(graph->node_list, 0, node_1);
  cth_array_set(CTHNode)(graph->node_list, 1, node_2);
  cth_array_set(CTHNode)(graph->node_list, 2, node_3);
  cth_array_set(CTHNode)(graph->node_list, 3, node_4);
  cth_array_set(CTHNode)(graph->node_list, 4, node_5);

  CTHConfig *config = (CTHConfig *)MALLOC(sizeof(CTHConfig));
  config->num_workers = 4;
  CTHScheduler *scheduler = cth_new_scheduler(config, graph);

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

  CTHList(CTHQueueJob) *ready_jobs = cth_new_list(CTHQueueJob)();
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

  CTHNode *node_1 = create_dummy_node(0, 0, 2);
  CTHNode *node_2 = create_dummy_node(1, 1, 1);
  CTHNode *node_3 = create_dummy_node(2, 1, 1);
  CTHNode *node_4 = create_dummy_node(3, 2, 1);
  CTHNode *node_5 = create_dummy_node(4, 1, 0);

  cth_array_set(CTHNode)(node_2->inbound_nodes, 0, node_1);
  cth_array_set(CTHNode)(node_3->inbound_nodes, 0, node_1);
  cth_array_set(CTHNode)(node_4->inbound_nodes, 0, node_2);
  cth_array_set(CTHNode)(node_4->inbound_nodes, 1, node_3);
  cth_array_set(CTHNode)(node_5->inbound_nodes, 0, node_4);

  CTHGraph *graph = create_dummy_graph(5);
  cth_array_set(CTHNode)(graph->node_list, 0, node_1);
  cth_array_set(CTHNode)(graph->node_list, 1, node_2);
  cth_array_set(CTHNode)(graph->node_list, 2, node_3);
  cth_array_set(CTHNode)(graph->node_list, 3, node_4);
  cth_array_set(CTHNode)(graph->node_list, 4, node_5);

  CTHConfig *config = (CTHConfig *)MALLOC(sizeof(CTHConfig));
  config->num_workers = CPU_CORES;
  CTHScheduler *scheduler = cth_new_scheduler(config, graph);
  CTHWorkerPool *pool = cth_new_pool(scheduler, config);
  cth_start_scheduler(scheduler);
  cth_close_pool(scheduler, pool);

  // all bits are good
  EXPECT_EQ(cth_are_all_bits_clear(scheduler->queue_status), true);
  EXPECT_EQ(cth_are_all_bits_clear(scheduler->ready_status), true);
  EXPECT_EQ(cth_are_all_bits_set(scheduler->done_status), true);

  for (list_index_t i = 0; i < scheduler->job_list->size; i++) {
    CTHQueueJob *job = cth_array_at(CTHQueueJob)(scheduler->job_list, i);
    EXPECT_EQ(job->status, CTH_JOB_STATUS_DONE);
  }
}

TEST(cTorchSchedulerTest, testManyTasks) {
  /**
   *  N nodes --> node_final
   */
  cth_array_index_t N_DEPENDENTS = 1000;

  CTHGraph *graph = create_dummy_graph(N_DEPENDENTS + 1);

  CTHNode *node_final = create_dummy_node(0, N_DEPENDENTS, 0);
  cth_array_set(CTHNode)(graph->node_list, N_DEPENDENTS, node_final);

  for (cth_array_index_t i = 0; i < N_DEPENDENTS; i++) {
    CTHNode *node = create_dummy_node(0, 0, 1);
    cth_array_set(CTHNode)(node_final->inbound_nodes, i, node);
    cth_array_set(CTHNode)(graph->node_list, i, node);
  }

  CTHConfig *config = (CTHConfig *)MALLOC(sizeof(CTHConfig));
  config->num_workers = CPU_CORES;
  CTHScheduler *scheduler = cth_new_scheduler(config, graph);
  CTHWorkerPool *pool = cth_new_pool(scheduler, config);
  cth_start_scheduler(scheduler);
  cth_close_pool(scheduler, pool);
}
