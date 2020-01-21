#include "backend.h"
#include "accelerate/accelerate.h"
#include "common.h"
#include "mkl/mkl.h"
#include "openBLAS/open_blas.h"

bool backend_support_op(CTH_OPERATOR_ID op_id, CTH_BACKEND backend) {
  // openBLAS should support all operators
  if (!openblas_supported(op_id))
    FAIL_EXIT("Unsupported operator.");

  if (backend == CTH_BACKEND_MKL) {
    return mkl_supported(op_id);
  } else if (backend == CTH_BACKEND_ACCELERATE) {
    return accl_supported(op_id);
  } else {
    // unrecognized backend
    FAIL_EXIT("Unsupported backend.");
  }
}

void dispatch_op_execution(CTorchOperator *op, CTH_BACKEND backend) {
  // Use fallback backend (openblas) if target backend does not support this op
  if (!backend_support_op(op->op_id, backend)) {
    backend = CTH_BACKEND_OPENBLAS;
  }

  switch (backend) {
  case CTH_BACKEND_ACCELERATE:
    accl_execute_op(op);
    break;
  case CTH_BACKEND_MKL:
    mkl_execute_op(op);
    break;
  default:
    openblas_execute_op(op);
    break;
  }
}