#include "cTorch/c_torch.h"
#include "tests/test_util.c"
#include "gtest/gtest.h"

TEST(absOpTest, testCpuX86Float) {
  uint32_t dims[] = {20, 10};
  CTorchNode *op_node = create_dummy_op_node(CTH_OP_ID_abs, dims);
  execute_node(op_node, CTH_BACKEND_CPU_X86);
  float *input =
      (float *)op_node->conent.op->in_bound_tensors->head->data->values;
  float *output =
      (float *)op_node->conent.op->out_bound_tensors->head->data->values;
  uint64_t n_ele =
      op_node->conent.op->in_bound_tensors->head->data->meta_info->n_elements;
  for (int i; i < n_ele; i++) {
    EXPECT_FLOAT_EQ(abs(input[i]), output[i]);
  }
}