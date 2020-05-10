#ifndef CTH_LOGGER_UTIL_H
#define CTH_LOGGER_UTIL_H

#include <stdbool.h>
#include <stdio.h>

/*******************************************************************************
 *
 * Util macros & functions
 */

/*
  Call FAIL_EXIT() if input pointer is NULL.
*/
#define FAIL_NULL_PTR(ptr)                                                     \
  do {                                                                         \
    if (ptr == NULL) {                                                         \
      FAIL_EXIT(CTH_LOG_ERR, "Pointer is NULL.");                              \
    }                                                                          \
  } while (0)

/*
  Print error message and exit.
*/
#define FAIL_EXIT(write_to, ...)                                               \
  do {                                                                         \
    char *tmp = NULL;                                                          \
    fprintf(stderr, "[cTorch][ERROR]: ");                                      \
    asprintf(&tmp, __VA_ARGS__);                                               \
    fprintf(stderr, "%s\n", tmp);                                              \
    free(tmp);                                                                 \
    exit(1);                                                                   \
  } while (0)

/*
  Log message. Auto break line at the end.

  Will check global config
*/
#define CTH_LOG(write_to, ...)                                                 \
  do {                                                                         \
    if (!CTH_LOG_ENABLE)                                                       \
      break;                                                                   \
                                                                               \
    char *tmp = NULL;                                                          \
    fprintf(stderr, "[cTorch][INFO]: ");                                       \
    asprintf(&tmp, __VA_ARGS__);                                               \
    fprintf(stderr, "%s\n", tmp);                                              \
    free(tmp);                                                                 \
  } while (0)

/*
  Exit if val is NAN.

  If global config CTH_NAN_EXIT is false, it will not exit.
*/
#define NAN_EXIT(val)                                                          \
  do {                                                                         \
    if (val != val && CTH_NAN_EXIT == true) {                                  \
      FAIL_EXIT(CTH_LOG_ERR, "Value is NaN");                                  \
    }                                                                          \
  } while (0)

/*******************************************************************************
 *
 * Global config
 */

/*
  Used by all log related functions.
*/
#define CTH_LOG_ERR 0;
#define CTH_LOG_INFO 1;

/*
  If true, execution will stop and exit if computation outputs NaN values.

  Default: false
*/
extern bool CTH_NAN_EXIT;

/*
  If true, LOG_F() will print out. Otherwise, ommit message.

  Default: true
*/
extern bool CTH_LOG_ENABLE;

#endif /* CTH_LOGGER_UTIL_H */
