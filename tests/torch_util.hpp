#include "cTorch/c_torch.h"
#include <torch/torch.h>

/**
 * @brief Create a torch tensor from a CTHTensor
 *
 * @param cth_tensor
 * @return torch::Tensor
 */
torch::Tensor create_torch_tensor(CTHTensor *cth_tensor);

void cp_values(torch::Tensor &py_tensor, CTHTensor *cth_tensor);

void print_pytensor(
    torch::Tensor &py_tensor,
    cth_tensor_dim_t n_ele,
    CTH_TENSOR_DATA_TYPE data_type);
