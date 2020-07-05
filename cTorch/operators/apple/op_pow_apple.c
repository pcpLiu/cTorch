#include "cTorch/operators/apple/op_list_apple.h"
#include "cTorch/operators/apple/util_apple.h"
#include <Accelerate/Accelerate.h>

/**
 * @brief pow(x, y)
 *
 * @note Apple backend only support float & double tensors
 *
 * Inputs & outputs:
 *    - # of input: 2
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_pow_apple(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *in_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int N = (int)in_1->meta_info->n_elements;
  // Accelerate use pow(in_2, in_1) order
  _cth_apple_vforce_function_call_binary(
      in_1->meta_info->data_type,
      pow,
      in_2->values,
      in_1->values,
      out->values,
      &N);
}
