#ifndef CTH_GRAPH_H
#define CTH_GRAPH_H

#include "cTorch/consts.h"
#include "cTorch/node.h"
#include <stdint.h>

/**
 * This struct represents a computational graph.
 */
typedef struct {
  char *graph_name;              /* Graph name. Optional */
  Array(CTorchNode) * node_list; /* Nodes containing in this graph */
} CTorchGraph;

#endif /* GRAPH_H */
