#include "cTorch/node.h"
#include "cTorch/logger_util.h"

impl_new_array_func(CTorchNode);
impl_array_at_func(CTorchNode);
impl_array_set_func(CTorchNode);

cth_impl_new_list_item_func(CTorchNode);
cth_impl_new_list_func(CTorchNode);
cth_impl_insert_list_func(CTorchNode);
cth_impl_list_contains_data_func(CTorchNode);
cth_impl_list_contains_item_func(CTorchNode);
cth_impl_list_at_func(CTorchNode);
cth_impl_list_pop_func(CTorchNode);
