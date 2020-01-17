#ifndef CTH_STORAGE_H
#define CTH_STORAGE_H

#include "common.h"
#include "consts.h"
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

typedef union {
  float *val_float;
  double *val_double;
  int8_t *val_int8;
} CTorchTensorValues;

typedef struct {
  CTorchTensorMeta meta_info;
  CTorchTensorValues values;
} CTorchTensor;

#endif /* STORAGE_H */
