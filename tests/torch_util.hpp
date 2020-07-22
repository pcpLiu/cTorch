#include "cTorch/c_torch.h"
#include <torch/torch.h>

/**
 * @brief Create a torch tensor from a CTorchTensor
 *
 * @param cth_tensor
 * @return torch::Tensor
 */
torch::Tensor create_torch_tensor(CTorchTensor *cth_tensor);
