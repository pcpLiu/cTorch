#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#ifndef ERFINV_PIl
/** The constant Pi in high precision */
#define ERFINV_PIl 3.1415926535897932384626433832795029L
#endif

#ifndef ERFINV_CONSTl
/** The constant used in _cth_erfinv_imp */
#define ERFINV_CONSTl 0.15449436008930206298828125L
#endif

/**
 * Ref: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
 */
long double _cth_erfinv_kernel(long double x) {
  long double tt1, tt2, lnx, sgn;
  sgn = (x < 0) ? -1.0f : 1.0f;

  x = (1 - x) * (1 + x); // x = 1 - x*x;
  lnx = logf(x);

  tt1 = 2 / (ERFINV_PIl * ERFINV_CONSTl) + 0.5f * lnx;
  tt2 = 1 / (ERFINV_CONSTl)*lnx;

  return (sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
}

// #define _cth_erfinv_kernel(input_ptr, output_ptr, N, data_type)                \
//   do {                                                                         \
//     data_type *input_t = (data_type *)input_ptr;                               \
//     data_type *output_t = (data_type *)output_ptr;                             \
//     for (int i = 0; i < N; i++) {                                              \
//       output_t[i] = (data_type)digammal(input_t[i]);                           \
//     }                                                                          \
//   } while (0)

/**
 * Computation inverse of erf
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_erfinv_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  tensor_size_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary(
      in->values, out->values, in->meta_info->data_type, N, _cth_erfinv_kernel);
}
