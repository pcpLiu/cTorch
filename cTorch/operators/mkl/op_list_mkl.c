#include "cTorch/operators/mkl/op_list_mkl.h"

// void (*fps_op_x86[ENABLED_OP_NUM])(CTorchOperator *) = {X86_ALL_OP_FUNCS};
void (*fps_op_mkl[ENABLED_OP_NUM])(CTorchOperator *) = {};
