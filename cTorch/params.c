#include "cTorch/params.h"

impl_new_array_func(CTorchParam);
impl_array_at_func(CTorchParam);
impl_array_set_func(CTorchParam);
impl_free_array_deep_func(CTorchParam);

void struct_deep_free(CTorchParam)(CTorchParam *param) {
  FAIL_NULL_PTR(param);
  FREE(param);
}

void cth_copy_param(CTorchParam *from_param, CTorchParam *to_param) {
  FAIL_NULL_PTR(from_param);
  FAIL_NULL_PTR(to_param);

  to_param->data = from_param->data;
  to_param->type = from_param->type;
}
