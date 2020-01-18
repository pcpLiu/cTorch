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
  struct ListTypeName(CTorchNode) * in_bound_nodes;
  struct ListTypeName(CTorchNode) * out_bound_nodes;

  CTorchNodeContent *conent;
} CTorchNode;

typedef ListStruct(CTorchNode) ListTypeName(CTorchNode);

declare_insert_func(CTorchNode, ListTypeName(CTorchNode),
                    ListInsertFuncName(CTorchNode));

// declare_insert_func(CTorchNode, ListTypeName(CTorchNode));
#endif /* NODE_H */
