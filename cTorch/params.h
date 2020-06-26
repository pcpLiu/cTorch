#ifndef CTH_PARAMS_H
#define CTH_PARAMS_H

#include "cTorch/generic_array.h"

/**
 * Enum used to indicate param type
 */
typedef enum CTH_PARAM_TYPE {
  CTH_PARAM_TYPE_MULTIPLIER_FLOAT32,
  CTH_PARAM_TYPE_MIN_FLOAT32,
  CTH_PARAM_TYPE_MAX_FLOAT32,
} CTH_PARAM_TYPE;

/**
 * Param data union
 */
typedef union CTorchParamData {
  /* multiplier */
  float multiplier;
  /* Min value threshold */
  float min;
  /* Max value threshold */
  float max;
} CTorchParamData;

typedef struct CTorchParam {
  CTH_PARAM_TYPE type;
  CTorchParamData data;
} CTorchParam;

// Array macros
def_array(CTorchParam);
declare_new_array_func(CTorchParam);
declare_array_at_func(CTorchParam);
declare_array_set_func(CTorchParam);

#endif /* PARAMS_H */
