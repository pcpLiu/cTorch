#include "cTorch/bit_array.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <string.h>
#include <tgmath.h>

/**
 * If size > 2^32, it beyond bit array's capability.
 * FAIL_EXIT()
 */
#define FORCE_BITS_SIZE(size, limit)                                           \
  {                                                                            \
    if (size > limit) {                                                        \
      FAIL_EXIT(CTH_LOG_ERR, "Bit size %d beyond 2^32 bits.", size);           \
    }                                                                          \
  }                                                                            \
  while (0)

bit_array_t *cth_new_bit_array(bit_array_index_t size) {
  bit_array_t *array = MALLOC(sizeof(bit_array_t));
  array->size = size;
  array->num_ints = ceil(size / BITS_INT_UNIT_SIZE);
  array->bits = MALLOC(sizeof(bit_array_int_t) * array->num_ints);
  memset(array->bits, 0, sizeof(bit_array_int_t) * array->num_ints);
  return array;
}

void cth_set_bit(bit_array_t *array, bit_array_index_t i) {
  FORCE_BITS_SIZE(i, array->size - 1);
  FAIL_NULL_PTR(array);

  bit_array_int_t flag = 1;
  flag = flag << (i % BITS_INT_UNIT_SIZE);
  *(array->bits + i / BITS_INT_UNIT_SIZE) |= flag;
}

void cth_clear_bit(bit_array_t *array, bit_array_index_t i) {
  FORCE_BITS_SIZE(i, array->size - 1);
  FAIL_NULL_PTR(array);

  bit_array_int_t flag = 1;
  flag = ~(flag << (i % BITS_INT_UNIT_SIZE));
  *(array->bits + i / BITS_INT_UNIT_SIZE) &= flag;
}

bool cth_is_bit_set(bit_array_t *array, bit_array_index_t i) {
  FORCE_BITS_SIZE(i, array->size - 1);
  FAIL_NULL_PTR(array);

  bit_array_int_t flag = 1;
  flag = flag << (i % BITS_INT_UNIT_SIZE);
  if (*(array->bits + i / BITS_INT_UNIT_SIZE) & flag) {
    return true;
  } else {
    return false;
  }
}