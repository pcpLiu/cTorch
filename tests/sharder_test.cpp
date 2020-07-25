#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

#include <cmath>

TEST(cTorchSharderTest, testTensorElewiseShardingMEMRECORD) {
  cth_tensor_dim_t n_dim = 3;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 7;
  dims[1] = 3;
  dims[2] = 20;
  CTHTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
  CTHList(CTHTensor) *shards = cth_new_list(CTHTensor)();
  cth_sharding_tensor_elewise(tensor, 7, shards);

  EXPECT_EQ(shards->size, 7);

  // check each shard tensor
  float *val_ptr = (float *)tensor->values;
  cth_tensor_dim_t ele_per_shard = 3 * 20;
  for (int shdard_i = 0; shdard_i < 7; shdard_i++) {
    CTHTensor *sharded_tensor = cth_list_at(CTHTensor)(shards, shdard_i);
    EXPECT_EQ(sharded_tensor->meta_info->n_elements, ele_per_shard);

    float *shard_ptr = (float *)sharded_tensor->values;
    float *val_ptr_offset = val_ptr + ele_per_shard * shdard_i;
    for (cth_tensor_dim_t ele_i = 0; ele_i < ele_per_shard; ele_i++) {
      EXPECT_EQ(*shard_ptr, *val_ptr_offset);
      shard_ptr++;
      val_ptr_offset++;
    }
  }

  // test in sing-thread mode
  struct_deep_free(CTHTensor)(tensor);
  cth_free_list_deep(CTHTensor)(shards);
  EXPECT_EQ(0, cth_get_num_unfree_records());
}

TEST(cTorchSharderTest, testOperatorElewiseShardingMEMRECORD) {
  cth_tensor_dim_t n_dim = 2;
  cth_tensor_dim_t *dims =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims[0] = 10;
  dims[1] = 20;
  CTHTensor *input = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  cth_tensor_dim_t *dims_2 =
      (cth_tensor_dim_t *)MALLOC(n_dim * sizeof(cth_tensor_dim_t));
  dims_2[0] = 8;
  dims_2[1] = 33;
  CTHTensor *output = create_dummy_tensor(
      dims_2, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0, 10.0);

  CTHOperator *op = create_dummy_op_with_param(CTH_OP_ID_abs, 1, 1, 0);
  cth_array_set(CTHTensor)(op->in_bound_tensors, 0, input);
  cth_array_set(CTHTensor)(op->out_bound_tensors, 0, output);

  CTHList(CTHOperator) *sharded_ops = cth_new_list(CTHOperator)();
  cth_sharding_op_elewise(op, 10, sharded_ops);

  EXPECT_EQ(10, sharded_ops->size);
  cth_tensor_dim_t n_ele_sum = 0;
  for (int i = 0; i < 10; i++) {
    CTHOperator *op = cth_list_at(CTHOperator)(sharded_ops, i);
    CTHTensor *input_tensor = cth_array_at(CTHTensor)(op->in_bound_tensors, 0);
    CTHTensor *output_tensor =
        cth_array_at(CTHTensor)(op->out_bound_tensors, 0);
    EXPECT_EQ(20, input_tensor->meta_info->n_elements);

    if (i != 9) {
      EXPECT_EQ((cth_tensor_dim_t)floor(output->meta_info->n_elements / 10),
                output_tensor->meta_info->n_elements);
    } else {
      EXPECT_EQ((cth_tensor_dim_t)floor(output->meta_info->n_elements / 10) +
                    output->meta_info->n_elements % 10,
                output_tensor->meta_info->n_elements);
    }
    n_ele_sum += output_tensor->meta_info->n_elements;
  }
  EXPECT_EQ(output->meta_info->n_elements, n_ele_sum);

  // test in sing-thread mode
  cth_free_list_deep(CTHOperator)(sharded_ops);
  struct_deep_free(CTHOperator)(op);
  EXPECT_EQ(0, cth_get_num_unfree_records());
}
