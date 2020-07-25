#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/operators/cuda/op_list_cuda.h"
#include "cTorch/operators/cuda/util_cuda.h"

_cth_declare_cuda_unary_kernel(cth_erf_cuda, erff, erf);

/**
 * @brief Calculate the error function of the input argument
 *
 * @param[CTHOperator] op operator
 *
 * @note CUDA onlly support float & double type
 *
 * Inputs & outputs:
 *   - # of input: 1
 *   - # of output: 1
 *   - Assume input & output have same types
 */
void op_erf_cuda(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  CTHTensor *input = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *output = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  CTH_TENSOR_DEVICE device = input->meta_info->device;

  _cth_cuda_unary_workflow(
      input->meta_info->data_type,
      input->values,
      output->values,
      input->meta_info->n_elements,
      CTH_CUDA_THREADS_PER_BLOCK,
      cth_erf_cuda,
      device);
}

#ifdef __cplusplus
}
#endif
