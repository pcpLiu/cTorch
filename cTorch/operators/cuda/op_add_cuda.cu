#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/operators/cuda/op_list_cuda.h"
#include "cTorch/operators/cuda/util_cuda.h"

#define _cth_add_cuda_atomic(x, y) (x + y)

_cth_declare_cuda_binary_kernel(
    cth_add_cuda, _cth_add_cuda_atomic, _cth_add_cuda_atomic);

/**
 * @brief Computes the element-wise acos value of the given input tensor
 *
 * @param[CTHOperator] op operator
 *
 * @note CUDA onlly support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 2
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_add_cuda(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 2, 1);
  CTHTensor *input_1 = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *input_2 = cth_array_at(CTHTensor)(op->in_bound_tensors, 1);
  CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DEVICE device = input_1->meta_info->device;

  _cth_cuda_binary_workflow(
      input_1->meta_info->data_type,
      input_1->values,
      input_2->values,
      output->values,
      input_1->meta_info->n_elements,
      CTH_CUDA_THREADS_PER_BLOCK,
      cth_add_cuda,
      device);
}

#ifdef __cplusplus
}
#endif