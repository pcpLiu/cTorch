#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/operators/cuda/op_list_cuda.h"
#include "cTorch/operators/cuda/util_cuda.h"

_cth_declare_cuda_binary_kernel(cth_pow_cuda, powf, pow);

/**
 * @brief Calculate the value of first argument to the power of second argument
 *
 * @param[CTorchOperator] op operator
 *
 * @note CUDA onlly support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 2
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_pow_cuda(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  CTorchTensor *input_1 = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *input_2 = array_at(CTorchTensor)(op->in_bound_tensors, 1);
  CTorchTensor *output = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DEVICE device = input_1->meta_info->device;

  _cth_cuda_binary_workflow(
      input_1->meta_info->data_type,
      input_1->values,
      input_2->values,
      output->values,
      input_1->meta_info->n_elements,
      CTH_CUDA_THREADS_PER_BLOCK,
      cth_pow_cuda,
      device);
}

#ifdef __cplusplus
}
#endif
