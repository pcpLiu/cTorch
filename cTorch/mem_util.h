#ifndef CTH_MEM_UTIL_H
#define CTH_MEM_UTIL_H

#include <stdlib.h>

#define MEMCPY memcpy

/**
 * @brief Malloc with optional argument to give a descriptive name to this
 * memory.
 *
 * @param size
 * @param record_name
 * @param file_name
 * @param line_num
 * @param func_name
 * @return void*
 */
void *cth_malloc(
    size_t size,
    const char *record_name,
    const char *file_name,
    int line_num,
    const char *func_name);

/**
 * Free pointer with some location info passed.
 */
void cth_free(
    void *ptr, const char *file_name, int lineno, const char *func_name);

/**
 * Free if pointer is not NULL. If it's NULL, do nothing.
 */
void cth_free_soft(
    void *ptr, const char *file_name, int lineno, const char *func_name);

/**
 * Just a wrapper of asprintf with memory record management for debug
 */
int cth_asprintf(char **strp, const char *fmt, ...);

/**
 * @brief malloc memory
 *
 * @note Inside this function, it calls calloc()
 *
 */
#define MALLOC(size) cth_malloc(size, NULL, __FILE__, __LINE__, __func__)

/**
 * @brief
 *
 */
#define MALLOC_NAME(size, name)                                                \
  cth_malloc(size, name, __FILE__, __LINE__, __func__)

/**
 * @brief
 *
 */
#define FREE(ptr) cth_free(ptr, __FILE__, __LINE__, __func__)

/**
 * @brief
 *
 */
#define FREE_SOFT(ptr) cth_free_soft(ptr, __FILE__, __LINE__, __func__)

#endif /* CTH_MEM_UTIL_H */
