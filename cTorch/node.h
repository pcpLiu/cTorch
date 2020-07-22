#ifndef CTH_NODE_H
#define CTH_NODE_H

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"
#include <stdlib.h>

typedef union {
  CTorchTensor *tensor;
  CTorchOperator *op;
} CTorchNodeContent;

/**
 * Node id type
 */
typedef uint32_t node_id_t;

/*
  CTorchNode.
  This struct represents a computational node in a torch graph.
  It could be either a operator or a tensor.
*/
typedef struct {
  node_id_t node_id;       /* Node id. Starting from 0 and consecutive */
  CTH_NODE_TYPE node_type; /* Node type: op or tensor */
  struct CTHArray(CTorchNode) * inbound_nodes;  /* Inbounds nodes */
  struct CTHArray(CTorchNode) * outbound_nodes; /* Inbounds nodes */
  CTorchNodeContent conent;                     /* Content */
} CTorchNode;

// Array macros
def_array(CTorchNode);
declare_new_array_func(CTorchNode);
declare_array_at_func(CTorchNode);
declare_array_set_func(CTorchNode);

// List macros
cth_def_list_item(CTorchNode);
def_list(CTorchNode);
cth_declare_new_list_item_func(CTorchNode);
cth_declare_new_list_func(CTorchNode);
cth_declare_insert_list_func(CTorchNode);
cth_declare_list_contains_data_func(CTorchNode);
cth_declare_list_contains_item_func(CTorchNode);
cth_declare_list_at_func(CTorchNode);
cth_declare_list_pop_func(CTorchNode);

#endif /* NODE_H */
