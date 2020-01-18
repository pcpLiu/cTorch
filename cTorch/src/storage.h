#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include "common.h"
#include "consts.h"
#include "list_d.h"
#include <stdint.h>
#include <uuid/uuid.h>

typedef struct {
  uuid_t uuid;
  uint8_t value_size_of;
  CTH_TENSOR_DATA_TYPE data_type;
  uint16_t n_dim;
  uint32_t *dim_size_list;
  CTH_TENSOR_TYPE type;
  /*
    For CTH_TENSOR_TYPE_PARAM type node, this is parameter name.
    As for other types, this is a optiona field and could ne null.
  */
  CTorchName *tensor_name;
} CTorchTensorMeta;

// Tensor value types
// ref: https://pytorch.org/docs/stable/tensors.html#torch-tensor
typedef union {
  float *val_float;
  double *val_double;
  int8_t *val_int8;
  uint8_t *val_uint8;
  int16_t *val_int16;
  int32_t *val_int32;
  int64_t *val_int64;
  bool *val_bool;
} CTorchTensorValues;

/*
  CTorchTensor.
  This struct represents a tensor obj in a computation gprah.
*/
typedef struct {
  CTorchTensorMeta meta_info;
  CTorchTensorValues values;
} CTorchTensor;

// List macro
def_list_item(CTorchTensor);
def_list(CTorchTensor);
declare_new_list_item_func(CTorchTensor);
declare_new_list_func(CTorchTensor);
declare_insert_list_func(CTorchTensor);
declare_list_contains_data_func(CTorchTensor);
declare_list_contains_item_func(CTorchTensor);

#endif /* STORAGE_H */
