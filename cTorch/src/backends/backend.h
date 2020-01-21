#ifndef CTH_BACKEND_H
#define CTH_BACKEND_H

#include "../operator.h"
#include "consts.h"

/*
  Check if target backend support this operator.
*/
bool backend_support_op(CTH_OPERATOR_ID, CTH_BACKEND);

/*
  Execute an operator with given backend
*/
void dispatch_op_execution(CTorchOperator *, CTH_BACKEND);

#endif /* BACKEND_H */
