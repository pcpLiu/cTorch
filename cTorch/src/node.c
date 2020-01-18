#include "node.h"
#include "common.h"
#include <stdbool.h>

impl_insert_func(CTorchNode, ListTypeName(CTorchNode),
                 ListInsertFuncName(CTorchNode));

impl_create_func(CTorchNode, ListTypeName(CTorchNode),
                 ListItemCreateFuncName(CTorchNode));

static CTorchNode *update_node_list(CTorchNode *target,
                                    ListTypeName(CTorchNode) * node_list,
                                    bool add_to_inbound) {
  FAIL_NULL_PTR(target);
  FAIL_NULL_PTR(node_list);

  ListTypeName(CTorchNode) *target_list =
      add_to_inbound ? target->inbound_nodes : target->outbound_nodes;

  while (node_list != NULL) {
    // reverse edge
    if (add_to_inbound) {
      ListInsertFuncName(CTorchNode)(node_list->data->outbound_nodes, target);
    } else {
      ListInsertFuncName(CTorchNode)(node_list->data->inbound_nodes, target);
    }
    // target edge
    ListInsertFuncName(CTorchNode)(target_list, node_list->data);
    node_list = node_list->next_item;
  }

  return target;
}

CTorchNode *c_torch_node_add_inbound_nodes(CTorchNode *target,
                                           ListTypeName(CTorchNode) *
                                               node_list) {
  return update_node_list(target, node_list, true);
}

CTorchNode *c_torch_node_add_outbound_nodes(CTorchNode *target,
                                            ListTypeName(CTorchNode) *
                                                node_list) {
  return update_node_list(target, node_list, false);
}