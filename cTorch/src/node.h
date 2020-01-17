#ifndef CTH_NODE_H
#define CTH_NODE_H

#include "consts.h"
#include "list.h"
#include "operator.h"
#include "storage.h"
#include <stdlib.h>

typedef union {
  CTorchTensor *tensor;
  CTorchOperator *op;
} CTorchNodeContent;

typedef struct {
  CTH_NODE_TYPE node_type;
  CTH_NODE_EXE_STATUS exe_status;

  /* Node list will be NULL if it's empty */
  struct ListCTorchNode *in_bound_nodes;
  struct ListCTorchNode *out_bound_nodes;

  CTorchNodeContent *conent;
} CTorchNode;

typedef ListStruct(CTorchNode) ListCTorchNode;

declare_insert_func(ListCTorchNode, CTorchNode);

#endif /* NODE_H */
