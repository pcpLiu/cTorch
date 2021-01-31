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
