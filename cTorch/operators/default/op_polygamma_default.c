#include "cTorch/operator.h"
#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

/**
 * @brief Computes the n^{th} derivative of the digamma function on input. n
 * \geq 0n≥0 is called the order of the polygamma function.
 *
 * @param op
 *
 * @todo Need implementation
 */
void op_polygamma_cpu(CTHOperator *op) {
  FAIL_EXIT(CTH_LOG_ERR, "op_polygamma_cpu not implemented");
}
