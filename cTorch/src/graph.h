#ifndef CTH_GRAPH_H
#define CTH_GRAPH_H

#include "common.h"
#include "consts.h"
#include "node.h"
#include <stdint.h>
#include <uuid/uuid.h>

/*
  CTorchGraph.
  This struct represents a computational graph.
*/
typedef struct {
  uuid_t uuid;
  CTorchName *graph_name; /* Optional */
  List(CTorchNode) * node_list;
} CTorchGraph;

// ListCTorchNode *c_torch_graph_input_nodes(CTorchGraph *);

#endif /* GRAPH_H */
