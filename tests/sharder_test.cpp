#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#include <cmath>

TEST(cTorchSharderTest, testTensorElewiseShardingMEMRECORD) {
  tensor_dim_t n_dim = 3;
  tensor_dim_t *dims = (tensor_dim_t *)MALLOC(n_dim * sizeof(tensor_dim_t));
  dims[0] = 7;
  dims[1] = 3;
  dims[2] = 20;
  CTorchTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
  List(CTorchTensor) *shards = new_list(CTorchTensor)();
  cth_sharding_tensor_elewise(tensor, 7, shards);

  EXPECT_EQ(shards->size, 7);

  // check each shard tensor
  float *val_ptr = (float *)tensor->values;
  tensor_size_t ele_per_shard = 3 * 20;
  for (int shdard_i = 0; shdard_i < 7; shdard_i++) {
    CTorchTensor *sharded_tensor = list_at(CTorchTensor)(shards, shdard_i);
    EXPECT_EQ(sharded_tensor->meta_info->n_elements, ele_per_shard);

    float *shard_ptr = (float *)sharded_tensor->values;
    float *val_ptr_offset = val_ptr + ele_per_shard * shdard_i;
    for (tensor_size_t ele_i = 0; ele_i < ele_per_shard; ele_i++) {
      EXPECT_EQ(*shard_ptr, *val_ptr_offset);
      shard_ptr++;
      val_ptr_offset++;
    }
  }

  // test in sing-thread mode
  struct_deep_free(CTorchTensor)(tensor);
  free_list_deep(CTorchTensor)(shards);
  EXPECT_EQ(0, cth_get_num_unfree_records());
}

TEST(cTorchSharderTest, testOperatorElewiseShardingMEMRECORD) {
  tensor_dim_t n_dim = 2;
  tensor_dim_t *dims = (tensor_dim_t *)MALLOC(n_dim * sizeof(tensor_dim_t));
  dims[0] = 10;
  dims[1] = 20;
  CTorchTensor *input = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  tensor_dim_t *dims_2 = (tensor_dim_t *)MALLOC(n_dim * sizeof(tensor_dim_t));
  dims_2[0] = 8;
  dims_2[1] = 33;
  CTorchTensor *output = create_dummy_tensor(
      dims_2, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  CTorchOperator *op = create_dummy_op(CTH_OP_ID_abs, 1, 1);
  array_set(CTorchTensor)(op->in_bound_tensors, 0, input);
  array_set(CTorchTensor)(op->out_bound_tensors, 0, output);

  List(CTorchOperator) *sharded_ops = new_list(CTorchOperator)();
  cth_sharding_op_elewise(op, 10, sharded_ops);

  EXPECT_EQ(10, sharded_ops->size);
  tensor_size_t n_ele_sum = 0;
  for (int i = 0; i < 10; i++) {
    CTorchOperator *op = list_at(CTorchOperator)(sharded_ops, i);
    CTorchTensor *input_tensor =
        array_at(CTorchTensor)(op->in_bound_tensors, 0);
    CTorchTensor *output_tensor =
        array_at(CTorchTensor)(op->out_bound_tensors, 0);
    EXPECT_EQ(20, input_tensor->meta_info->n_elements);

    if (i != 9) {
      EXPECT_EQ((tensor_size_t)floor(output->meta_info->n_elements / 10),
                output_tensor->meta_info->n_elements);
    } else {
      EXPECT_EQ((tensor_size_t)floor(output->meta_info->n_elements / 10) +
                    output->meta_info->n_elements % 10,
                output_tensor->meta_info->n_elements);
    }
    n_ele_sum += output_tensor->meta_info->n_elements;
  }
  EXPECT_EQ(output->meta_info->n_elements, n_ele_sum);

  // test in sing-thread mode
  free_list_deep(CTorchOperator)(sharded_ops);
  struct_deep_free(CTorchOperator)(op);
  EXPECT_EQ(0, cth_get_num_unfree_records());
}
