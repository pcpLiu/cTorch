#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * Add two input tensors and store result in output tensor. No broadcast
 * support.
 *
 * Assume input and output tensor have same datatypes. If not, this op will
 * fail.
 */
void op_add_cpu(CTorchOperator *op) {}