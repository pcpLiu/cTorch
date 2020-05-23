#ifndef CTH_GRAPH_H
#define CTH_GRAPH_H

#include "cTorch/consts.h"
#include "cTorch/node.h"
#include <stdint.h>
#include <uuid/uuid.h>

/*
  CTorchGraph.
  This struct represents a computational graph.
*/
typedef struct {
  // UUID of this graph
  uuid_t uuid;

  // Graph name. Optional
  char *graph_name;

  // Nodes containing in this graph
  List(CTorchNode) * node_list;
} CTorchGraph;

#endif /* GRAPH_H */
