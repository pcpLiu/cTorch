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
  CTH_PARAM_TYPE_P_FLOAT32,
  CTH_PARAM_TYPE_DIM_INT32,
} CTH_PARAM_TYPE;

/**
 * Param data union
 */
typedef union CTHParamData {
  float multiplier;
  float min;
  float max;
  float p;
  int32_t dim;
} CTHParamData;

typedef struct CTHParam {
  CTH_PARAM_TYPE type;
  CTHParamData data;
} CTHParam;

// Array macros
cth_def_array(CTHParam);
cth_declare_new_array_func(CTHParam);
cth_declare_array_at_func(CTHParam);
cth_declare_array_set_func(CTHParam);
cth_declare_free_array_deep_func(CTHParam);

/**
 * Deep free a CTHParam
 *
 * Note:
 *    If pointer is NULL, error raised and exit.
 */
void struct_deep_free(CTHParam)(CTHParam *param);

/**
 * Copy fields from `from_param` to `to_param`
 */
void cth_copy_param(CTHParam *from_param, CTHParam *to_param);

/**
 * Extract param value with given types
 */
#define EXTRACT_PARAM_VALUE(op, param_type, param_data_field, param_var)       \
  do {                                                                         \
    CTHParam *param = cth_get_param_by_type(op, param_type, true);             \
    param_var =                                                                \
        cth_get_param_by_type(op, param_type, true)->data.param_data_field;    \
  } while (0)

#endif /* PARAMS_H */
