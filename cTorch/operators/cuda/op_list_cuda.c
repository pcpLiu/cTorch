#include "cTorch/operators/cuda/op_list_cuda.h"

// void (*fps_op_x86[ENABLED_OP_NUM])(CTHOperator *) = {X86_ALL_OP_FUNCS};
void (*fps_op_cuda[ENABLED_OP_NUM])(CTHOperator *) = {};
