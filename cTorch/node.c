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

#include "cTorch/node.h"
#include "cTorch/logger_util.h"

cth_impl_new_array_func(CTHNode);
cth_impl_array_at_func(CTHNode);
cth_impl_array_set_func(CTHNode);

cth_impl_new_list_item_func(CTHNode);
cth_impl_new_list_func(CTHNode);
cth_impl_insert_list_func(CTHNode);
cth_impl_list_contains_data_func(CTHNode);
cth_impl_list_contains_item_func(CTHNode);
cth_impl_list_at_func(CTHNode);
cth_impl_list_pop_func(CTHNode);
