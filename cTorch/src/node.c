#include "node.h"

impl_insert_func(CTorchNode, ListTypeName(CTorchNode),
                 ListInsertFuncName(CTorchNode));

impl_create_func(CTorchNode, ListTypeName(CTorchNode),
                 ListItemCreateFuncName(CTorchNode));