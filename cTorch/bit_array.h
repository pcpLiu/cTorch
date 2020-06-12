#ifndef CTH_BIT_ARRAY_H
#define CTH_BIT_ARRAY_H

#include <stdbool.h>
#include <stdint.h>

/**
 * Single integer bits size
 */
#define BITS_INT_UNIT_SIZE 32

typedef uint32_t bit_array_index_t;
typedef uint32_t bit_array_int_t;

/**
 * A bit array to represent true/false information in an effecient way.
 *
 * Max # of bits to store: 2^32
 */
typedef struct bit_array_t {
  bit_array_int_t *bits;      /* Integer array to store the bits */
  bit_array_index_t num_ints; /* Integer array size */
  bit_array_index_t size; /* How many bits store in this array (logically) */
} bit_array_t;

/**
 * Create a new bit array with targed size
 */
bit_array_t *cth_new_bit_array(bit_array_index_t size);

/**
 * Set bit at index i to 1
 */
void cth_set_bit(bit_array_t *array, bit_array_index_t i);

/**
 * Set bit at index i to 0
 */
void cth_clear_bit(bit_array_t *array, bit_array_index_t i);

/**
 * Check if bit at index i is 1
 */
bool cth_is_bit_set(bit_array_t *array, bit_array_index_t i);

#endif /* BIT_ARRAY_H */
