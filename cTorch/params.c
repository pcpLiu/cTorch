#include "cTorch/params.h"

cth_impl_new_array_func(CTHParam);
cth_impl_array_at_func(CTHParam);
cth_impl_array_set_func(CTHParam);
cth_impl_free_array_deep_func(CTHParam);

void struct_deep_free(CTHParam)(CTHParam *param) {
  FAIL_NULL_PTR(param);
  FREE(param);
}

void cth_copy_param(CTHParam *from_param, CTHParam *to_param) {
  FAIL_NULL_PTR(from_param);
  FAIL_NULL_PTR(to_param);

  to_param->data = from_param->data;
  to_param->type = from_param->type;
}
