#ifndef CTH_PARAMS_H
#define CTH_PARAMS_H

#include "cTorch/consts.h"
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
declare_free_array_deep_func(CTorchParam);

/**
 * Deep free a CTorchParam
 *
 * Note:
 *    If pointer is NULL, error raised and exit.
 */
void struct_deep_free(CTorchParam)(CTorchParam *param);

/**
 * Copy fields from `from_param` to `to_param`
 */
void cth_copy_param(CTorchParam *from_param, CTorchParam *to_param);

/**
 * Extract param value with given types
 */
#define EXTRACT_PARAM_VALUE(op, param_type, param_data_field, param_var)       \
  do {                                                                         \
    CTorchParam *param = cth_get_param_by_type(op, param_type, true);          \
    param_var =                                                                \
        cth_get_param_by_type(op, param_type, true)->data.param_data_field;    \
  } while (0)

#endif /* PARAMS_H */
