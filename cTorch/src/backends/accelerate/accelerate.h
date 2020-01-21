#ifndef CTH_ACCELERATE_H
#define CTH_ACCELERATE_H

#include "../../consts.h"

bool accl_supported(CTH_OPERATOR_ID);

void accl_execute_op(CTorchOperator *);

#endif /* ACCELERATE_H */
