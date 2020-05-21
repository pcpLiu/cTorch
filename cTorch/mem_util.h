#ifndef CTH_MEM_UTIL_H
#define CTH_MEM_UTIL_H

#include <stdlib.h>

/**
 * Malloc with optional argument to give a descriptive name to this memory.
 *
 * Parameters:
 *    - size: memory size
 *    - name: memory block name. Could be NULL
 */
void *cth_malloc(size_t size, const char *name);

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

#define MALLOC(size) cth_malloc(size, NULL)
#define MALLOC_NAME(size, name) cth_malloc(size, name)
#define FREE(ptr) cth_free(ptr, __FILE__, __LINE__, __func__)
#define FREE_SOFT(ptr) cth_free_soft(ptr, __FILE__, __LINE__, __func__)
#define MEMCPY memcpy

#endif /* CTH_MEM_UTIL_H */
