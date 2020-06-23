#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_addcmul(op, data_type)                                            \
  do {                                                                         \
    CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);     \
    CTorchTensor *tensor_1 = get_input_by_name(op, "tensor_1", true);          \
    CTorchTensor *tensor_2 = get_input_by_name(op, "tensor_2", true);          \
    CTorchTensor *tensor_value = get_input_by_name(op, "value", true);         \
    float value = ((float *)tensor_value->values)[0];                          \
    CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);   \
    tensor_size_t N = tensor_1->meta_info->n_elements;                         \
                                                                               \
    for (tensor_size_t i = 0; i < N; i++) {                                    \
      ((data_type *)output->values)[i] =                                       \
          ((data_type *)input->values)[i] +                                    \
          value * ((data_type *)tensor_1->values)[i] *                         \
              ((data_type *)tensor_2->values)[i];                              \
    }                                                                          \
  } while (0);

/**
 * out = input + value * tensor_1 * tensor_2)
 * ref: https://pytorch.org/docs/stable/torch.html#torch.addcmul
 *
 * Op requirement:
 *    - # of input tensors: 4
 *        - input: the input always at index 0
 *        - tensor_1: tensor with name `tensor_1`
 *        - tensor_2: tensor with name `tensor_2`
 *        - value: tensor with name `value`, float type, scalar
 *    - # of output tensors: 1
 *        - output:  the input always at index 0
 *
 * Note: does not support boradcast
 */
void op_addcmul_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 4, 1);
  CTorchTensor *input = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  _cpu_generic_compute(op, _cth_addcmul, input->meta_info->data_type);
}
