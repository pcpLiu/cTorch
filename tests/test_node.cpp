#include "c_torch.h"
#include "gtest/gtest.h"

TEST(cTorchNodeTest, test_add_bound_nodes) {
  // target
  CTorchNode a = {
      .node_type = CTH_NODE_TYPE_DATA,
      .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
      .inbound_nodes = NewList(ListTypeName(CTorchNode)),
      .outbound_nodes = NewList(ListTypeName(CTorchNode)),
  };

  // node list
  CTorchNode b = {
      .node_type = CTH_NODE_TYPE_DATA,
      .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
      .inbound_nodes = NewList(ListTypeName(CTorchNode)),
      .outbound_nodes = NewList(ListTypeName(CTorchNode)),
  };
  CTorchNode c = {
      .node_type = CTH_NODE_TYPE_DATA,
      .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
      .inbound_nodes = NewList(ListTypeName(CTorchNode)),
      .outbound_nodes = NewList(ListTypeName(CTorchNode)),
  };
  ListTypeName(CTorchNode) *node_list = NewList(ListTypeName(CTorchNode));
  node_list->data = &b;
  // ListInsertFuncName(CTorchNode)(node_list, &c);

  // c_torch_node_add_inbound_nodes(&a, node_list);
}