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