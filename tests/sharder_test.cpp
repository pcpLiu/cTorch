#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchSharderTest, testTensorElewiseSharding) {
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

  // Memory check
  data_deep_free(CTorchTensor)(tensor);
  free_list_deep(CTorchTensor)(shards);
  EXPECT_EQ(0, cth_get_num_unfree_records());
  cth_print_unfree_records();
}