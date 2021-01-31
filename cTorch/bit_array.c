/**
 * Copyright 2021 Zhonghao Liu
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

cth_bit_array_t *cth_new_bit_array(cth_bit_cth_array_index_t size) {
  FORCE_NOT_EQ(0, size, "bit array size could not be 0");

  cth_bit_array_t *array = MALLOC(sizeof(cth_bit_array_t));
  array->size = size;
  array->num_ints = 1 + ((size - 1) / BITS_INT_UNIT_SIZE); // size != 0
  array->bits = MALLOC(sizeof(cth_bit_array_int_t) * array->num_ints);

  /* Clear all bits */
  for (cth_bit_cth_array_index_t i = 0; i < array->num_ints; i++) {
    *(array->bits + i) = 0;
  }
  return array;
}

void cth_set_bit(cth_bit_array_t *array, cth_bit_cth_array_index_t i) {
  FORCE_BITS_SIZE(i, array->size - 1);
  FAIL_NULL_PTR(array);

  cth_bit_array_int_t flag = 1;
  flag = flag << (i % BITS_INT_UNIT_SIZE);
  *(array->bits + i / BITS_INT_UNIT_SIZE) |= flag;
}

void cth_clear_bit(cth_bit_array_t *array, cth_bit_cth_array_index_t i) {
  FORCE_BITS_SIZE(i, array->size - 1);
  FAIL_NULL_PTR(array);

  cth_bit_array_int_t flag = 1;
  flag = ~(flag << (i % BITS_INT_UNIT_SIZE));
  *(array->bits + i / BITS_INT_UNIT_SIZE) &= flag;
}

bool cth_is_bit_set(cth_bit_array_t *array, cth_bit_cth_array_index_t i) {
  FORCE_BITS_SIZE(i, array->size - 1);
  FAIL_NULL_PTR(array);

  cth_bit_array_int_t flag = 1;
  flag = flag << (i % BITS_INT_UNIT_SIZE);
  if (*(array->bits + i / BITS_INT_UNIT_SIZE) & flag) {
    return true;
  } else {
    return false;
  }
}

bool cth_are_all_bits_clear(cth_bit_array_t *array) {
  FAIL_NULL_PTR(array);

  /**
   * All integers' values are 0
   */
  for (cth_bit_cth_array_index_t i = 0; i < array->num_ints; i++) {
    if (*(array->bits + i) != 0) {
      return false;
    }
  }

  return true;
}

bool cth_are_all_bits_set(cth_bit_array_t *array) {
  FAIL_NULL_PTR(array);

  /**
   * First n-1 integers' values are 2^32.
   */
  cth_bit_array_int_t all_one = ~(0 & 0);
  if (array->num_ints > 1) {
    for (cth_bit_cth_array_index_t i = 0; i < array->num_ints - 1; i++) {
      if (*(array->bits + i) != all_one) {
        return false;
      }
    }
  }

  /**
   * Last integer, bits in range [size % 32: 0] should 1. Like:
   *    0...000011111...11
   */
  cth_bit_array_int_t mod_bits = array->size % BITS_INT_UNIT_SIZE;
  cth_bit_array_int_t mask = (1 << mod_bits) - 1; /* 0...000011111...11 */
  cth_bit_array_int_t last_int = *(array->bits + array->num_ints - 1);
  return last_int == mask;
}
