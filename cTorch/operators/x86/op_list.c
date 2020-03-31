#include "cTorch/operators/x86/op_list.h"

// void (*fps_op_x86[ENABLED_OP_NUM])(CTorchOperator *) = {X86_ALL_OP_FUNCS};
void (*fps_op_x86[ENABLED_OP_NUM])(CTorchOperator *) = {};