#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/*
  This operator will braodcast input tensor to target dimension and store values
  to output tenso.

  Assume works on one pair of input & output tensors.

  Required a param tensor: `target_dims`.

  Broadcasting semantics:
  https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
*/
void op_broadcast_cpu(CTorchOperator *op) {
  FORCE_OP_PARAM_EXIST(op, "target_dims", CTH_TENSOR_DATA_TYPE_UINT_32);
  // CTorchTensor *target_dims = ;

  CTorchTensor *in = op->in_bound_tensors->head->data;
  CTorchTensor *out = op->in_bound_tensors->head->data;
  // FORCE_TENSOR_DIMENSION(out, )
}