/**
 * Copyright 2021 Zhonghao Liu
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#ifndef CTH_ERFINV_CONSTS
#define erfinv_a3 -0.140543331
#define erfinv_a2 0.914624893
#define erfinv_a1 -1.645349621
#define erfinv_a0 0.886226899

#define erfinv_b4 0.012229801
#define erfinv_b3 -0.329097515
#define erfinv_b2 1.442710462
#define erfinv_b1 -2.118377725
#define erfinv_b0 1

#define erfinv_c3 1.641345311
#define erfinv_c2 3.429567803
#define erfinv_c1 -1.62490649
#define erfinv_c0 -1.970840454

#define erfinv_d2 1.637067800
#define erfinv_d1 3.543889200
#define erfinv_d0 1
#endif /* CTH_ERFINV_CONSTS */

/**
 * @brief Implementation was borrowed from libit (http://libit.sourceforge.net/)
 *
 * TODO: libit holds a GPL license which is not fully compatible with MIT.
 * We may need re-implement this by ourselves.
 *
 */
double _cth_erfinv_kernel(double x) {
  double x2, r, y;
  int sign_x;

  if (x < -1 || x > 1)
    return NAN;

  if (x == 0)
    return 0;

  if (x > 0)
    sign_x = 1;
  else {
    sign_x = -1;
    x = -x;
  }

  if (x <= 0.7) {

    x2 = x * x;
    r = x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
    r /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 + erfinv_b1) *
             x2 +
         erfinv_b0;
  } else {
    y = sqrt(-log((1 - x) / 2));
    r = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
    r /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
  }

  r = r * sign_x;
  x = x * sign_x;

  r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r));
  r -= (erf(r) - x) / (2 / sqrt(M_PI) * exp(-r * r));

  return r;
}

/**
 * Computation inverse of erf
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 *    - Input and output should be same dimention and type.
 */
void op_erfinv_cpu(CTHOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTHTensor *in = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
  CTHTensor *out = cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
  cth_tensor_dim_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary(
      in->values, out->values, in->meta_info->data_type, N, _cth_erfinv_kernel);
}
