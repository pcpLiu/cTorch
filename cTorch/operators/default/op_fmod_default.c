#include <tgmath.h>

#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * fmod element wise.
 *
 * Op requirement:
 *    - # of input tensors: 2
 *        - 0: dividend
 *        - 1: divisor
 *    - # of output tensors: 1
 *        - output:  the input always at index 0
 */
void op_fmod_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in_a = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *in_b = array_at(CTorchTensor)(op->in_bound_tensors, 1);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  int64_t N = in_a->meta_info->n_elements;
  _cpu_1d_map_elewise_binary(
      in_a->values,
      in_b->values,
      out->values,
      in_a->meta_info->data_type,
      N,
      fmod);
}
