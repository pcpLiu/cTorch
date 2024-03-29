#include "tests/torch_util.hpp"
#include "tests/test_util.h"

/**
 * @brief Get the dim from tensor object
 *
 * @return std::vector<int64_t>
 */
std::vector<int64_t> *get_dim_from_tensor(CTHTensor *cth_tensor) {
  auto shape = new std::vector<int64_t>();
  for (cth_tensor_dim_t i = 0; i < cth_tensor->meta_info->n_dim; i++) {
    shape->push_back((cth_tensor_dim_t)cth_tensor->meta_info->dims[i]);
  }
  return shape;
}

/**
 * @brief Set the dtype based on cth tensor type
 *
 * @param py_tensor
 * @param cth_tensor
 */
torch::TensorOptions *make_options(CTHTensor *cth_tensor) {
  auto options = new torch::TensorOptions();
  options->requires_grad(false);

  // options->layout(torch::kStrided);
  switch (cth_tensor->meta_info->data_type) {
  case CTH_TENSOR_DATA_TYPE_BOOL:
    options->dtype(torch::kBool);
    break;
  case CTH_TENSOR_DATA_TYPE_FLOAT_32:
    options->dtype(torch::kFloat32);
    break;
  case CTH_TENSOR_DATA_TYPE_FLOAT_16:
    options->dtype(torch::kFloat32);
    break;
  case CTH_TENSOR_DATA_TYPE_FLOAT_64:
    options->dtype(torch::kFloat64);
    break;
  case CTH_TENSOR_DATA_TYPE_INT_16:
    options->dtype(torch::kInt16);
    break;
  case CTH_TENSOR_DATA_TYPE_INT_32:
    options->dtype(torch::kInt32);
    break;
  case CTH_TENSOR_DATA_TYPE_INT_64:
    options->dtype(torch::kInt64);
    break;
  case CTH_TENSOR_DATA_TYPE_UINT_8:
    options->dtype(torch::kUInt8);
    break;
  default:
    FAIL_EXIT(CTH_LOG_ERR, "Unsupported types");
    break;
  }
  return options;
}

#define _fill_value(py_tensor, cth_values_ptr, i)                              \
  do {                                                                         \
    py_tensor.index_put_(                                                      \
        at::indexing::TensorIndex{(int64_t)i}, cth_values_ptr[i]);             \
  } while (0)

/**
 * @brief Cp values
 *
 * @param py_tensor
 * @param cth_tensor
 */
void cp_values(torch::Tensor &py_tensor, CTHTensor *cth_tensor) {
  CTH_TENSOR_DATA_TYPE data_type = cth_tensor->meta_info->data_type;
  for (cth_tensor_dim_t i = 0; i < cth_tensor->meta_info->n_elements; i++) {
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
        data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
      _fill_value(py_tensor, ((float *)cth_tensor->values), i);
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
      _fill_value(py_tensor, ((double *)cth_tensor->values), i);
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
      _fill_value(py_tensor, ((int16_t *)cth_tensor->values), i);
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
      _fill_value(py_tensor, ((int32_t *)cth_tensor->values), i);
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
      _fill_value(py_tensor, ((int64_t *)cth_tensor->values), i);
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
      _fill_value(py_tensor, ((uint8_t *)cth_tensor->values), i);
    } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
      _fill_value(py_tensor, ((bool *)cth_tensor->values), i);
    }
  }
}

torch::Tensor create_torch_tensor(CTHTensor *cth_tensor) {
  auto options = make_options(cth_tensor);
  auto py_tensor = torch::zeros(cth_tensor->meta_info->n_elements, *options);
  cp_values(py_tensor, cth_tensor);

  auto shape = get_dim_from_tensor(cth_tensor);
  auto ret_tensor = py_tensor.reshape(*shape);
  return std::move(ret_tensor);

  // auto options = make_options(cth_tensor);
  // auto shape = get_dim_from_tensor(cth_tensor);
  // auto tensor = torch::from_blob((float *)cth_tensor->values, *shape);
  // return std::move(tensor);
}

void print_pytensor(
    torch::Tensor &py_tensor,
    cth_tensor_dim_t n_ele,
    CTH_TENSOR_DATA_TYPE data_type) {
  std::cout << "[";
  auto pytorch_result_tensor_flat = py_tensor.reshape({n_ele});
  for (cth_tensor_dim_t i = 0; i < n_ele; i++) {
    if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
        data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
      std::cout << pytorch_result_tensor_flat[i].item<float>() << ", ";
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_8) {
      std::cout << pytorch_result_tensor_flat[i].item<int8_t>() << ", ";
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
      std::cout << pytorch_result_tensor_flat[i].item<int16_t>() << ", ";
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
      std::cout << pytorch_result_tensor_flat[i].item<int32_t>() << ", ";
    } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
      std::cout << pytorch_result_tensor_flat[i].item<int64_t>() << ", ";
    } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
      std::cout << pytorch_result_tensor_flat[i].item<uint8_t>() << ", ";
    } else if (data_type == CTH_TENSOR_DATA_TYPE_BOOL) {
      std::cout << pytorch_result_tensor_flat[i].item<bool>() << ", ";
    } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
      std::cout << pytorch_result_tensor_flat[i].item<double>() << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
