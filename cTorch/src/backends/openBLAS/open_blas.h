#ifndef CTH_OPEN_BLAS_H
#define CTH_OPEN_BLAS_H

#include "../../consts.h"
#include "../../operator.h"

bool openblas_supported(CTH_OPERATOR_ID);

void openblas_execute_op(CTorchOperator *);

#endif /* OPEN_BLAS_H */
