#include "cTorch/operators/default/op_list.h"

// void (*fps_op_x86[ENABLED_OP_NUM])(CTorchOperator *) = {X86_ALL_OP_FUNCS};
void (*fps_op_default[ENABLED_OP_NUM])(CTorchOperator *) = {op_abs_cpu,
                                                            op_acos_cpu};