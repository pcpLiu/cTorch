#include "cTorch/operators/apple/op_list_apple.h"
#include "cTorch/operators/apple/util_apple.h"
#include <Accelerate/Accelerate.h>

/**
 * @brief round(x)
 *
 * @note Apple backend only support float & double tensors
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_round_apple(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int N = (int)in->meta_info->n_elements;
  _cth_apple_vforce_function_call_unary(
      in->meta_info->data_type, nint, in->values, out->values, &N);
}
