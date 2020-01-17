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
  CTorchNodeList *in_bound_nodes;
  CTorchNodeList *out_bound_nodes;

  CTorchNodeContent *conent;
} CTorchNode;

typedef ListStruct(CTorchNode) CTorchNodeList;

declare_insert_func(CTorchNodeList, CTorchNode)

#endif /* NODE_H */
