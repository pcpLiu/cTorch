#ifndef CTH_NODE_H
#define CTH_NODE_H

#include "cTorch/consts.h"
#include "cTorch/generic_array.h"
#include "cTorch/operator.h"
#include "cTorch/storage.h"
#include <stdlib.h>

typedef union {
  CTHTensor *tensor;
  CTHOperator *op;
} CTHNodeContent;

/**
 * Node id type
 */
typedef uint32_t node_id_t;

/*
  CTHNode.
  This struct represents a computational node in a torch graph.
  It could be either a operator or a tensor.
*/
typedef struct {
  node_id_t node_id;       /* Node id. Starting from 0 and consecutive */
  CTH_NODE_TYPE node_type; /* Node type: op or tensor */
  struct CTHArray(CTHNode) * inbound_nodes;  /* Inbounds nodes */
  struct CTHArray(CTHNode) * outbound_nodes; /* Inbounds nodes */
  CTHNodeContent conent;                     /* Content */
} CTHNode;

// Array macros
cth_def_array(CTHNode);
cth_declare_new_array_func(CTHNode);
cth_declare_array_at_func(CTHNode);
cth_declare_array_set_func(CTHNode);

// List macros
cth_def_list_item(CTHNode);
def_list(CTHNode);
cth_declare_new_list_item_func(CTHNode);
cth_declare_new_list_func(CTHNode);
cth_declare_insert_list_func(CTHNode);
cth_declare_list_contains_data_func(CTHNode);
cth_declare_list_contains_item_func(CTHNode);
cth_declare_list_at_func(CTHNode);
cth_declare_list_pop_func(CTHNode);

#endif /* NODE_H */
