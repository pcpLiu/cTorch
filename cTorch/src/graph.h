#ifndef CTH_GRAPH_H
#define CTH_GRAPH_H

#include "common.h"
#include "consts.h"
#include "node.h"
#include <stdint.h>
#include <uuid/uuid.h>

typedef struct {
  uuid_t uuid;
  CTorchName *graph_name;
  ListCTorchNode *nodes;
} CTorchGraph;

ListCTorchNode *c_torch_graph_input_nodes(CTorchGraph *);

#endif /* GRAPH_H */
