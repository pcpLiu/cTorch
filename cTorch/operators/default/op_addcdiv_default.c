#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#define _cth_addcdiv(op, data_type)                                            \
  do {                                                                         \
    CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);       \
    CTHTensor *tensor_1 = cth_get_input_by_name(op, "tensor_1", false);        \
    CTHTensor *tensor_2 = cth_get_input_by_name(op, "tensor_2", false);        \
    CTHParam *param =                                                          \
        cth_get_param_by_type(op, CTH_PARAM_TYPE_MULTIPLIER, true);            \
    float multiplier = *(param->data.multiplier);                              \
    CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);     \
    cth_tensor_dim_t N = tensor_1->meta_info->n_elements;                      \
                                                                               \
    for (cth_tensor_dim_t i = 0; i < N; i++) {                                 \
      ((data_type *)output->values)[i] =                                       \
          ((data_type *)input->values)[i] +                                    \
          multiplier * (((data_type *)tensor_1->values)[i] /                   \
                        ((data_type *)tensor_2->values)[i]);                   \
    }                                                                          \
  } while (0);

/**
 * out = input + multiplier * (tensor_1 / tensor_2)
 * ref: https://pytorch.org/docs/stable/torch.html#torch.addcdiv
 *
 * Op requirement:
 *    - # of input tensors: 4
 *        - input: the input always at index 0
 *        - tensor_1: tensor with name `tensor_1`
 *        - tensor_2: tensor with name `tensor_2`
 *    - # of arguments: 1
 *        - CTH_PARAM_TYPE_MULTIPLIER
 *    - # of output tensors: 1
 *        - output:  the input always at index 0
 *
 * Note: does not support boradcast
 */
void op_addcdiv_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 3, 1);
  FORCE_OP_PARAM_NUM(op, 1);

  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTH_TENSOR_DATA_TYPE data_type = input->meta_info->data_type;
  FORCE_OP_INPUT_EXIST(op, "tensor_1", data_type);
  FORCE_OP_INPUT_EXIST(op, "tensor_2", data_type);
  FORCE_OP_PARAM_EXIST(op, CTH_PARAM_TYPE_MULTIPLIER);

  _cpu_generic_compute(op, _cth_addcdiv, data_type);
}
