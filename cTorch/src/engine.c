#include "engine.h"
#include "backends/backend.h"

/*
  For data (tensor) node, we just mark its status and do nothing else.
*/
void execute_node(CTorchNode *node, CTH_BACKEND backend) {
  if (node->node_type == CTH_NODE_TYPE_OPERATOR) {
    dispatch_op_execution((CTorchOperator *)node->conent, backend);
  }
  node->exe_status = CTH_NODE_EXE_STATUS_DIRTY;
}
