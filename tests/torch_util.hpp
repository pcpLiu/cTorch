#include "cTorch/c_torch.h"
#include <torch/torch.h>

/**
 * @brief Create a torch tensor from a CTHTensor
 *
 * @param cth_tensor
 * @return torch::Tensor
 */
torch::Tensor create_torch_tensor(CTHTensor *cth_tensor);
