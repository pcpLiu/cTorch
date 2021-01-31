#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "tests/torch_util.hpp"
#include "gtest/gtest.h"

torch::Tensor
_ReplicationPad3d_pytorch(torch::Tensor &pytorch_in_tensor, CTHOperator *op) {

  cth_pad_t *padding;
  EXTRACT_PARAM_VALUE(op, CTH_PARAM_TYPE_PADDING_D6, padding, padding);
  cth_tensor_dim_t padding_left = padding[0];
  cth_tensor_dim_t padding_right = padding[1];
  cth_tensor_dim_t padding_top = padding[2];
  cth_tensor_dim_t padding_bottom = padding[3];
  cth_tensor_dim_t padding_front = padding[4];
  cth_tensor_dim_t padding_back = padding[5];
  auto m = torch::nn::ReplicationPad3d(torch::nn::ReplicationPad3dOptions(
      {padding_left,
       padding_right,
       padding_top,
       padding_bottom,
       padding_front,
       padding_back}));
  return m(pytorch_in_tensor);
}

void test_replication_pad_3d(
    CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type, float min, float max) {
  CTHNode *op_node = create_dummy_op_node_unary_3d_padding(
      CTH_OP_ID_ReplicationPad3d, data_type, min, max);
  CTHOperator *op = op_node->conent.op;

  if (backend == CTH_BACKEND_DEFAULT) {
    op_ReplicationPad3d_cpu(op);
  }

  if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
      data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, float, EXPECT_EQ_PRECISION, 1e-3, _ReplicationPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, double, EXPECT_EQ_PRECISION, 1e-3, _ReplicationPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_UINT_8) {
    _ele_wise_equal_nn_op_pytorch(
        op, uint8_t, EXPECT_EQ_PRECISION, 1e-3, _ReplicationPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
    _ele_wise_equal_nn_op_pytorch(
        op, int16_t, EXPECT_EQ_PRECISION, 1e-3, _ReplicationPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
    _ele_wise_equal_nn_op_pytorch(
        op, int32_t, EXPECT_EQ_PRECISION, 1e-3, _ReplicationPad3d_pytorch);
  } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
    _ele_wise_equal_nn_op_pytorch(
        op, int64_t, EXPECT_EQ_PRECISION, 1e-3, _ReplicationPad3d_pytorch);
  }
}

TEST(cTorchReplicationPad3dOpTest, testFloat16Default) {
  test_replication_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0, 100.0);
}

TEST(cTorchReplicationPad3dOpTest, testFloat32Default) {
  test_replication_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 100.0);
}

TEST(cTorchReplicationPad3dOpTest, testFloat64Default) {
  test_replication_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0, 100.0);
}

TEST(cTorchReplicationPad3dOpTest, testUInt8Default) {
  test_replication_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_UINT_8, 1.0, 100.0);
}

TEST(cTorchReplicationPad3dOpTest, testInt16Default) {
  test_replication_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0, 100.0);
}

TEST(cTorchReplicationPad3dOpTest, testInt32Default) {
  test_replication_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0, 100.0);
}

TEST(cTorchReplicationPad3dOpTest, testInt64Default) {
  test_replication_pad_3d(
      CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0, 100.0);
}
