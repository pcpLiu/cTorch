#include "cTorch/engine.h"

#define BACKEND_MISSING_ERR_MSG(backend)                                       \
  "Try to execute operator on " #backend                                       \
  "backend. But cTorch was not built for this backend."

/*
  Dispatch operator execution based on target backend.

  Note:
   If the backend was not built at compiling time, cTorch directly exits.
*/
void dispatch_op_execution(CTorchOperator *op, CTH_BACKEND backend) {
  CTH_OP_ID op_id = op->op_id;
  if (backend == CTH_BACKEND_CPU_X86) {
#ifdef BACKEND_CPU_X86
    FAIL_NULL_PTR(fps_op_x86[CTH_OP_ID_abs]);
    (*fps_op_x86[op_id])(op);
#else
    FAIL_EXIT(BACKEND_MISSING_ERR_MSG(x86));
#endif
  } else if (backend == CTH_BACKEND_CPU_ARM) {
#ifdef BACKEND_CPU_ARM
    FAIL_NULL_PTR(fps_op_x86[op_id]);
    (*fps_op_arm[op_id])(op);
#else
    FAIL_EXIT(BACKEND_MISSING_ERR_MSG(ARM));
#endif
  } else if (backend == CTH_BACKEND_MKL) {
#ifdef BACKEND_MKL
    FAIL_NULL_PTR(fps_op_x86[op_id]);
    (*fps_op_mkl[op_id])(op);
#else
    FAIL_EXIT(BACKEND_MISSING_ERR_MSG(MKL));
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    FAIL_NULL_PTR(fps_op_x86[op_id]);
    (*fps_op_cuda[op_id])(op);
#else
    FAIL_EXIT(BACKEND_MISSING_ERR_MSG(CUDA));
#endif
  } else if (backend == CTH_BACKEND_OPENBLAS) {
#ifdef BACKEND_OPENBLAS
    FAIL_NULL_PTR(fps_op_x86[op_id]);
    (*fps_op_openblas[op_id])(op);
#else
    FAIL_EXIT(BACKEND_MISSING_ERR_MSG(openBLAS));
#endif
  } else if (backend == CTH_BACKEND_ACCELERATE) {
#ifdef BACKEND_ACCELERATE
    FAIL_NULL_PTR(fps_op_x86[op_id]);
    (*fps_op_accelerate[op_id])(op);
#else
    FAIL_EXIT(BACKEND_MISSING_ERR_MSG(Accelerate));
#endif
  }
}

/*
  Execute a node.
    - Operator node: execute the computation
    - Data node: only mark its status and do nothing else
*/
void execute_node(CTorchNode *node, CTH_BACKEND backend) {
  if (node->node_type == CTH_NODE_TYPE_OPERATOR) {
    dispatch_op_execution(node->conent.op, backend);
  }
  node->exe_status = CTH_NODE_EXE_STATUS_DIRTY;
}
