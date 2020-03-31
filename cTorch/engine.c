#include "cTorch/engine.h"

#define _BACKEND_MISSING_ERR_MSG(backend)                                      \
  "Try to execute operator on " #backend                                       \
  "backend. But cTorch was not built for this backend."

#define _BACKEND_FALLBACK_EXE(op_fps)                                          \
  {                                                                            \
    if (op_fps[op_id] == NULL) {                                               \
      fall_back = true;                                                        \
    } else {                                                                   \
      (*op_fps[op_id])(op);                                                    \
    }                                                                          \
  }

/*
  Dispatch operator execution based on target backend.

  Note:
   If the backend was not built at compiling time, cTorch directly exits.
*/
void dispatch_op_execution(CTorchOperator *op, CTH_BACKEND backend) {
  CTH_OP_ID op_id = op->op_id;
  bool fall_back = false;
  if (backend == CTH_BACKEND_CPU_X86) {
#ifdef BACKEND_CPU_X86
    _BACKEND_FALLBACK_EXE(fps_op_x86);
#else
    FAIL_EXIT(CTH_LOG_STR, _BACKEND_MISSING_ERR_MSG(x86));
#endif
  } else if (backend == CTH_BACKEND_CPU_ARM) {
#ifdef BACKEND_CPU_ARM
    _BACKEND_FALLBACK_EXE(fps_op_arm);
#else
    FAIL_EXIT(CTH_LOG_STR, _BACKEND_MISSING_ERR_MSG(ARM));
#endif
  } else if (backend == CTH_BACKEND_MKL) {
#ifdef BACKEND_MKL
    _BACKEND_FALLBACK_EXE(fps_op_mkl);
#else
    FAIL_EXIT(CTH_LOG_STR, _BACKEND_MISSING_ERR_MSG(MKL));
#endif
  } else if (backend == CTH_BACKEND_CUDA) {
#ifdef BACKEND_CUDA
    _BACKEND_FALLBACK_EXE(fps_op_cuda);
#else
    FAIL_EXIT(CTH_LOG_STR, _BACKEND_MISSING_ERR_MSG(CUDA));
#endif
  } else if (backend == CTH_BACKEND_OPENBLAS) {
#ifdef BACKEND_OPENBLAS
    _BACKEND_FALLBACK_EXE(fps_op_openblas);
#else
    FAIL_EXIT(CTH_LOG_STR, _BACKEND_MISSING_ERR_MSG(openBLAS));
#endif
  } else if (backend == CTH_BACKEND_ACCELERATE) {
#ifdef BACKEND_ACCELERATE
    _BACKEND_FALLBACK_EXE(fps_op_openblas);
#else
    FAIL_EXIT(CTH_LOG_STR, _BACKEND_MISSING_ERR_MSG(Accelerate));
#endif
  }

  if (fall_back || backend == CTH_BACKEND_DEFAULT) {
    if (fps_op_default[op_id] == NULL) {
      FAIL_EXIT(CTH_LOG_STR, "Unsupported Operator ID %d.", op_id);
    } else {
      (*fps_op_default[op_id])(op);
    }
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
