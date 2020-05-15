#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

TEST(cTorchSharderTest, testTensorElewiseSharding) {
  tensor_dim_t dims[] = {7, 3, 20};
  tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
  CTorchTensor *tensor = create_dummy_tensor(
      dims, n_dim, CTH_TENSOR_DATA_TYPE_FLOAT_32, -1.0, 1.0);
  List(CTorchTensor) *shards = new_list(CTorchTensor)();
  cth_sharding_tensor_elewise(tensor, 7, shards);

  EXPECT_EQ(shards->size, 7);

  float *val_ptr = (float *)tensor->values;
  tensor_size_t ele_per_shard = 3 * 20;
  for (int shdard_i = 0; shdard_i < 7; shdard_i++) {
    CTorchTensor *sharded_tensor = list_at(CTorchTensor)(shards, shdard_i);
    float *shard_ptr = (float *)sharded_tensor->values;
    float *val_ptr_offset = val_ptr + ele_per_shard * shdard_i;
    // check each element
    for (tensor_size_t ele_i = 0; ele_i < ele_per_shard; ele_i++) {
      EXPECT_EQ(*shard_ptr, *val_ptr_offset);
      shard_ptr++;
      val_ptr_offset++;
    }
  }
}