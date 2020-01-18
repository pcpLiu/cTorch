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
  struct ListTypeName(CTorchNode) * inbound_nodes;
  struct ListTypeName(CTorchNode) * outbound_nodes;

  CTorchNodeContent *conent;
} CTorchNode;

typedef ListStruct(CTorchNode) ListTypeName(CTorchNode);

declare_insert_func(CTorchNode, ListTypeName(CTorchNode),
                    ListInsertFuncName(CTorchNode));

declare_create_func(CTorchNode, ListTypeName(CTorchNode),
                    ListItemCreateFuncName(CTorchNode));

/*
  Add a list of nodes into target node's in/out-bound list.
  Side effect:
    - These two functions will automatically update list nodes'
    in/out-bound infor.
*/
CTorchNode *c_torch_node_add_inbound_nodes(CTorchNode *,
                                           ListTypeName(CTorchNode) *);
CTorchNode *c_torch_node_add_outbound_nodes(CTorchNode *,
                                            ListTypeName(CTorchNode) *);
#endif /* NODE_H */
