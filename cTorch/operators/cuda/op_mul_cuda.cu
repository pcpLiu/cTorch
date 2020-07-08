#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/operators/cuda/op_list_cuda.h"
#include "cTorch/operators/cuda/util_cuda.h"

#define _cth_mul_cuda_atomic(x, y) (x * y)

_cth_declare_cuda_binary_kernel(
    cth_mul_cuda, _cth_mul_cuda_atomic, _cth_mul_cuda_atomic);

/**
 * @brief Each element of the tensor input is multiplied by the corresponding
 * element ofhe Tensor other. The resulting tensor is returned.
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
void op_mul_cuda(CTorchOperator *op) {
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
      cth_mul_cuda,
      device);
}

#ifdef __cplusplus
}
#endif
