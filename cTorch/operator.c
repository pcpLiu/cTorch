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

#include "cTorch/operator.h"
#include "cTorch/consts.h"
#include "cTorch/debug_util.h"
#include "cTorch/logger_util.h"

#include <string.h>

cth_impl_new_list_item_func(CTHOperator);
cth_impl_new_list_func(CTHOperator);
cth_impl_insert_list_func(CTHOperator);
cth_impl_list_at_func(CTHOperator);
cth_impl_list_pop_func(CTHOperator);
cth_impl_free_list_func(CTHOperator);
cth_impl_free_list_deep_func(CTHOperator);

void FORCE_INPUT_OUTPUT_TSR_NUM_EQ(CTHOperator *op) {
  FAIL_NULL_PTR(op);

  if (op->in_bound_tensors->size != op->out_bound_tensors->size) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "Operator should have same numbers of input and output tensors.");
  }
}

void OP_FAIL_ON_DTYPE(CTHOperator *op, CTH_TENSOR_DATA_TYPE data_type) {
  FAIL_NULL_PTR(op);

  CTHTensor *tensor = NULL;
  for (cth_array_index_t i = 0; i < op->in_bound_tensors->size; i++) {
    tensor = cth_array_at(CTHTensor)(op->in_bound_tensors, i);
    if (tensor->meta_info->data_type == data_type) {
      FAIL_EXIT(CTH_LOG_ERR, "Operator does not support data type.");
    }
  }

  for (cth_array_index_t i = 0; i < op->out_bound_tensors->size; i++) {
    tensor = cth_array_at(CTHTensor)(op->in_bound_tensors, i);
    if (tensor->meta_info->data_type == data_type) {
      FAIL_EXIT(CTH_LOG_ERR, "Operator does not support data type.");
    }
  }
}

void FORCE_OP_INPUT_EXIST(
    CTHOperator *op, const char *name, CTH_TENSOR_DATA_TYPE data_type) {
  FAIL_NULL_PTR(op);

  CTHTensor *tensor = NULL;
  bool found = false;
  for (cth_array_index_t i = 0; i < op->in_bound_tensors->size; i++) {
    tensor = cth_array_at(CTHTensor)(op->in_bound_tensors, i);
    if (strcmp(name, tensor->meta_info->tensor_name) == 0 &&
        tensor->meta_info->data_type == data_type) {
      found = true;
      break;
    }
  }

  if (!found) {
    FAIL_EXIT(CTH_LOG_ERR, "FORCE_OP_INPUT_EXIST fails");
  }
}

void FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(
    const CTHOperator *op,
    const cth_array_index_t num_input,
    const cth_array_index_t num_output) {
  FAIL_NULL_PTR(op);

  if (num_input != op->in_bound_tensors->size) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "FORCE_OP_INPUT_OUTPUT_TENSOR_NUM failS. Required No. of input tensors "
        "is %u, given %u",
        num_input,
        op->in_bound_tensors->size);
  }

  if (num_output != op->out_bound_tensors->size) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "FORCE_OP_INPUT_OUTPUT_TENSOR_NUM fails. Required No. of output "
        "tensors "
        "is %u, given %u",
        num_output,
        op->out_bound_tensors->size);
  }
}

void FORCE_OP_PARAM_NUM(
    const CTHOperator *op, const cth_array_index_t num_param) {
  FAIL_NULL_PTR(op);

  if (op->params->size != num_param) {
    FAIL_EXIT(
        CTH_LOG_ERR,
        "FORCE_OP_PARAM_NUM fails. Required No. of output params is %d, op has "
        "%d",
        num_param,
        op->out_bound_tensors->size);
  }
}

void FORCE_OP_PARAM_EXIST(CTHOperator *op, const CTH_PARAM_TYPE type) {
  FAIL_NULL_PTR(op);
  FAIL_NULL_PTR(op->params);
  for (cth_array_index_t i = 0; i < op->params->size; i++) {
    CTHParam *param = cth_array_at(CTHParam)(op->params, i);
    FAIL_NULL_PTR(param);
    if (type == param->type) {
      return;
    }
  }

  FAIL_EXIT(CTH_LOG_ERR, "FORCE_OP_PARAM_EXIST failed.");
}

CTHTensor *_get_tensor_by_name(
    CTHArray(CTHTensor) * tensor_array, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(tensor_array);
  CTHTensor *target_tensor = NULL;
  for (cth_array_index_t i = 0; i < tensor_array->size; i++) {
    CTHTensor *tensor = cth_array_at(CTHTensor)(tensor_array, i);
    if (strcmp(tensor->meta_info->tensor_name, name) == 0) {
      target_tensor = tensor;
      break;
    }
  }

  if (target_tensor == NULL && fail_exit) {
    FAIL_EXIT(CTH_LOG_ERR, "Could not find tensor %s", name);
  } else {
    return target_tensor;
  }
}

CTHTensor *
cth_get_input_by_name(CTHOperator *op, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(op);
  return _get_tensor_by_name(op->in_bound_tensors, name, true);
}

CTHTensor *
get_output_by_name(CTHOperator *op, const char *name, bool fail_exit) {
  FAIL_NULL_PTR(op);
  return _get_tensor_by_name(op->out_bound_tensors, name, true);
}

void struct_deep_free(CTHOperator)(CTHOperator *op) {
  cth_free_array_deep(CTHTensor)(op->in_bound_tensors);
  cth_free_array_deep(CTHTensor)(op->out_bound_tensors);
  cth_free_array_deep(CTHParam)(op->params);
  FREE(op);
}

CTHParam *cth_get_param_by_type(
    const CTHOperator *op, const CTH_PARAM_TYPE type, bool fail_exit) {
  FAIL_NULL_PTR(op);

  for (int i = 0; i < op->params->size; i++) {
    CTHParam *param = cth_array_at(CTHParam)(op->params, i);
    FAIL_NULL_PTR(param);
    if (type == param->type) {
      return param;
    }
  }

  if (fail_exit) {
    FAIL_EXIT(CTH_LOG_ERR, "Cannot find param with type %u", type);
  } else {
    return NULL;
  }
}

void cth_extract_param_value(
    const CTHOperator *op,
    CTH_PARAM_TYPE param_type,
    void **param_var,
    bool fail_on_null) {
  CTHParam *param = cth_get_param_by_type(op, param_type, fail_on_null);

  if (param == NULL) {
    *param_var = NULL;
    return;
  }

  if (param_type == CTH_PARAM_TYPE_MULTIPLIER) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_MIN) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_MAX) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_P) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_DIM) {
    *param_var = param->data.dim_val;
  } else if (param_type == CTH_PARAM_TYPE_IN_CHANNELS) {
    *param_var = param->data.dim_val;
  } else if (param_type == CTH_PARAM_TYPE_OUT_CHANNELS) {
    *param_var = param->data.dim_val;
  } else if (param_type == CTH_PARAM_TYPE_GROUPS) {
    *param_var = param->data.dim_val;
  } else if (param_type == CTH_PARAM_TYPE_KERNEL_SIZE) {
    *param_var = param->data.dim_val;
  } else if (param_type == CTH_PARAM_TYPE_STRIDE) {
    *param_var = param->data.dim_val;
  } else if (param_type == CTH_PARAM_TYPE_DILATION) {
    *param_var = param->data.dim_val;
  } else if (param_type == CTH_PARAM_TYPE_KERNEL_SIZE_D2) {
    *param_var = param->data.dim_2_val;
  } else if (param_type == CTH_PARAM_TYPE_STRIDE_D2) {
    *param_var = param->data.dim_2_val;
  } else if (param_type == CTH_PARAM_TYPE_DILATION_D2) {
    *param_var = param->data.dim_2_val;
  } else if (param_type == CTH_PARAM_TYPE_PADDING_D2) {
    *param_var = param->data.dim_2_val;
  } else if (param_type == CTH_PARAM_TYPE_PADDING_D4) {
    *param_var = param->data.dim_4_val;
  } else if (param_type == CTH_PARAM_TYPE_PADDING_D6) {
    *param_var = param->data.dim_6_val;
  } else if (param_type == CTH_PARAM_TYPE_PADDING_MODE) {
    *param_var = param->data.padding_mode_val;
  } else if (param_type == CTH_PARAM_TYPE_PADDING_VALUE_FLOAT) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_ALPHA_FLOAT) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_LAMBD_FLOAT) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_NEGATIVE_SLOPE_FLOAT) {
    *param_var = param->data.float_val;
  } else if (param_type == CTH_PARAM_TYPE_NUM_PARAMETERS) {
    *param_var = param->data.dim_val;
  } else {
    FAIL_EXIT(CTH_LOG_ERR, "Uknown parameter type: %d", param_type);
  }
}
