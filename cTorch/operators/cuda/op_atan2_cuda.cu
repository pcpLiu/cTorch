#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/operators/cuda/op_list_cuda.h"
#include "cTorch/operators/cuda/util_cuda.h"

#define atan2f_block(in_1, in_2, out, i)                                       \
  do {                                                                         \
    out_ptr_d[i] = atan2f(in_ptr_2_d[i], in_ptr_1_d[i]);                       \
  } while (0)

#define atan2d_block(in_1, in_2, out, i)                                       \
  do {                                                                         \
    out_ptr_d[i] = atan2(in_ptr_2_d[i], in_ptr_1_d[i]);                        \
  } while (0)

_cth_declare_cuda_binary_kernel_generic(
    cth_atan2_cuda, atan2f_block, atan2d_block);

/**
 * @brief Calculate the arc tangent of the ratio of first and second input
 * arguments
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
void op_atan2_cuda(CTHOperator *op) {
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
      cth_atan2_cuda,
      device);
}

#ifdef __cplusplus
}
#endif
