#ifndef CTH_NODE_H
#define CTH_NODE_H

#include "cTorch/consts.h"
#include "cTorch/list_d.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"
#include <stdlib.h>
#include <uuid/uuid.h>

typedef union {
  CTorchTensor *tensor;
  CTorchOperator *op;
} CTorchNodeContent;

/*
  CTorchNode.
  This struct represents a computational node in a torch graph.
  It could be either a operator or a tensor.
*/
typedef struct {
  // UUID of this node
  uuid_t uuid;

  // node type
  CTH_NODE_TYPE node_type;

  /* Node list will be NULL if it's empty */
  struct List(CTorchNode) * inbound_nodes;
  struct List(CTorchNode) * outbound_nodes;

  // A tensor or operator
  CTorchNodeContent conent;
} CTorchNode;

// List macros
def_list_item(CTorchNode);
def_list(CTorchNode);
declare_new_list_item_func(CTorchNode);
declare_new_list_func(CTorchNode);
declare_insert_list_func(CTorchNode);
declare_list_contains_data_func(CTorchNode);
declare_list_contains_item_func(CTorchNode);
declare_list_at_func(CTorchNode);

/**
 *
 */
CTorchNode *
cth_add_inbound_nodes(CTorchNode *target_node, List(CTorchNode) * nodes);

/**
 *
 */
CTorchNode *
cth_add_outbound_nodes(CTorchNode *target_node, List(CTorchNode) * nodes);

#endif /* NODE_H */
