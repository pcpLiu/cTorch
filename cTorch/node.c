#include "cTorch/node.h"
#include "cTorch/common.h"

impl_new_list_item_func(CTorchNode);
impl_new_list_func(CTorchNode);
impl_insert_list_func(CTorchNode);
impl_list_contains_data_func(CTorchNode);
impl_list_contains_item_func(CTorchNode);

static CTorchNode *update_node_list(
    CTorchNode *target, List(CTorchNode) * node_list, bool add_to_inbound) {
  FAIL_NULL_PTR(target);
  FAIL_NULL_PTR(node_list);
  if (node_list->size == 0) {
    FAIL_EXIT(CTH_LOG_ERR, "node_list is empty");
  }

  List(CTorchNode) *target_list =
      add_to_inbound ? target->inbound_nodes : target->outbound_nodes;

  ListItem(CTorchNode) *item = node_list->head;
  while (item != NULL) {
    // check duplicate
    if (list_contains_data(CTorchNode)(target_list, item->data) != NULL) {
      continue;
    }

    // reverse edge
    if (add_to_inbound) {
      insert_list(CTorchNode)(item->data->outbound_nodes, target);
    } else {
      insert_list(CTorchNode)(item->data->inbound_nodes, target);
    }

    // target edge
    insert_list(CTorchNode)(target_list, item->data);
    item = item->next_item;
  }

  return target;
}

CTorchNode *c_torch_node_add_inbound_nodes(
    CTorchNode *target, List(CTorchNode) * node_list) {
  return update_node_list(target, node_list, true);
}

CTorchNode *c_torch_node_add_outbound_nodes(
    CTorchNode *target, List(CTorchNode) * node_list) {
  return update_node_list(target, node_list, false);
}