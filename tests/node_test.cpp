#include "cTorch/c_torch.h"
#include "gtest/gtest.h"

CTorchNode new_dum_node() {
  CTorchNode node = {
    node_type : CTH_NODE_TYPE_DATA,
    exe_status : CTH_NODE_EXE_STATUS_CLEAN,
    inbound_nodes : new_list(CTorchNode)(),
    outbound_nodes : new_list(CTorchNode)(),
  };
  return node;
}

TEST(cTorchNodeTest, testAddBoundNodes) {
  // target
  CTorchNode a = new_dum_node();

  // Add to inbound
  CTorchNode b = new_dum_node();
  CTorchNode c = new_dum_node();
  List(CTorchNode) *inbound_list = new_list(CTorchNode)();
  insert_list(CTorchNode)(inbound_list, &b);
  insert_list(CTorchNode)(inbound_list, &c);

  c_torch_node_add_inbound_nodes(&a, inbound_list);
  ASSERT_EQ(a.inbound_nodes->size, 2);

  ListItem(CTorchNode) *item_b =
      list_contains_data(CTorchNode)(a.inbound_nodes, &b);
  ASSERT_NE(item_b, nullptr);
  EXPECT_EQ(item_b->data, &b);
  ListItem(CTorchNode) *item_a =
      list_contains_data(CTorchNode)(b.outbound_nodes, &a);
  ASSERT_NE(item_a, nullptr);
  EXPECT_EQ(item_a->data, &a);

  ListItem(CTorchNode) *item_c =
      list_contains_data(CTorchNode)(a.inbound_nodes, &c);
  ASSERT_NE(item_c, nullptr);
  EXPECT_EQ(item_c->data, &c);
  item_a = list_contains_data(CTorchNode)(c.outbound_nodes, &a);
  ASSERT_NE(item_a, nullptr);
  EXPECT_EQ(item_a->data, &a);

  // Add to outbound
  CTorchNode d = new_dum_node();
  CTorchNode e = new_dum_node();
  List(CTorchNode) *outbound_list = new_list(CTorchNode)();
  insert_list(CTorchNode)(outbound_list, &d);
  insert_list(CTorchNode)(outbound_list, &e);

  c_torch_node_add_outbound_nodes(&a, outbound_list);
  ASSERT_EQ(a.outbound_nodes->size, 2);

  ListItem(CTorchNode) *item_d =
      list_contains_data(CTorchNode)(a.outbound_nodes, &d);
  ASSERT_NE(item_d, nullptr);
  EXPECT_EQ(item_d->data, &d);
  item_a = list_contains_data(CTorchNode)(d.inbound_nodes, &a);
  ASSERT_NE(item_a, nullptr);
  EXPECT_EQ(item_a->data, &a);

  ListItem(CTorchNode) *item_e =
      list_contains_data(CTorchNode)(a.outbound_nodes, &e);
  ASSERT_NE(item_e, nullptr);
  EXPECT_EQ(item_e->data, &e);
  item_a = list_contains_data(CTorchNode)(e.inbound_nodes, &a);
  ASSERT_NE(item_a, nullptr);
  EXPECT_EQ(item_a->data, &a);
}