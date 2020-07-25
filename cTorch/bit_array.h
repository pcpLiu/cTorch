#ifndef CTH_BIT_ARRAY_H
#define CTH_BIT_ARRAY_H

#include <stdbool.h>
#include <stdint.h>

/**
 * Single integer bits size
 */
#define BITS_INT_UNIT_SIZE 32

typedef uint32_t cth_bit_cth_array_index_t;
typedef uint32_t cth_bit_array_int_t;

/**
 * A bit array to represent true/false information in an effecient way.
 *
 * Max # of bits to store: 2^32
 */
typedef struct cth_bit_array_t {
  cth_bit_array_int_t *bits;          /* Integer array to store the bits */
  cth_bit_cth_array_index_t num_ints; /* Integer array size */
  cth_bit_cth_array_index_t
      size; /* How many bits store in this array (logically) */
} cth_bit_array_t;

/**
 * Create a new bit array with targed size. When created, all bits are cleared.
 */
cth_bit_array_t *cth_new_bit_array(cth_bit_cth_array_index_t size);

/**
 * Set bit at index i to 1
 */
void cth_set_bit(cth_bit_array_t *array, cth_bit_cth_array_index_t i);

/**
 * Set bit at index i to 0
 */
void cth_clear_bit(cth_bit_array_t *array, cth_bit_cth_array_index_t i);

/**
 * Check if bit at index i is 1
 */
bool cth_is_bit_set(cth_bit_array_t *array, cth_bit_cth_array_index_t i);

/**
 * Check if all logical bits in this array are set.
 */
bool cth_are_all_bits_set(cth_bit_array_t *array);

/**
 * Check if all logical bits in this array are cleared.
 */
bool cth_are_all_bits_clear(cth_bit_array_t *array);

#endif /* BIT_ARRAY_H */
