#ifndef CTH_MKL_H
#define CTH_MKL_H

#include "../../consts.h"

bool mkl_supported(CTH_OPERATOR_ID);

void mkl_execute_op(CTorchOperator *);

#endif /* MKL_H */
