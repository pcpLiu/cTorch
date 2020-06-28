#ifndef CTH_LOGGER_UTIL_H
#define CTH_LOGGER_UTIL_H

#include <stdbool.h>
#include <stdio.h>

/*******************************************************************************
 *
 * Util macros & functions
 */

#define _FAIL_NULL_PTR(ptr, file_name, line_num, func_name, extra_msg)         \
  do {                                                                         \
    if (ptr == NULL) {                                                         \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR,                                                         \
          "Pointer is NULL. (file: %s, line: %d, function: %s, extra msg: "    \
          "%s)",                                                               \
          file_name,                                                           \
          line_num,                                                            \
          func_name,                                                           \
          extra_msg);                                                          \
    }                                                                          \
  } while (0)

#define FAIL_NULL_PTR(ptr)                                                     \
  _FAIL_NULL_PTR(ptr, __FILE__, __LINE__, __func__, "N/A")

#define FAIL_NULL_PTR_MSG(ptr, msg)                                            \
  _FAIL_NULL_PTR(ptr, __FILE__, __LINE__, __func__, msg)

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
    const char *out_str = (write_to == stderr ? "ERROR" : "INFO");             \
    fprintf(write_to, "[cTorch][%s]: ", out_str);                              \
                                                                               \
    char *tmp = NULL;                                                          \
    if (-1 == asprintf(&tmp, __VA_ARGS__)) {                                   \
      fprintf(write_to, "asprintf() call failed!\n");                          \
    } else {                                                                   \
      fprintf(write_to, "%s\n", tmp);                                          \
      free(tmp);                                                               \
    }                                                                          \
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

/**
 * Force a == b. This assumes a, b are primiteve types.
 */
#define FORCE_EQ(a, b, ...)                                                    \
  do {                                                                         \
    if (a != b) {                                                              \
      FAIL_EXIT(CTH_LOG_ERR, "FORCE_EQ failes. Info: %s", __VA_ARGS__);        \
    }                                                                          \
  } while (0)

/**
 * Force a != b. This assumes a, b are primiteve types.
 */
#define FORCE_NOT_EQ(a, b, ...)                                                \
  do {                                                                         \
    if (a == b) {                                                              \
      FAIL_EXIT(CTH_LOG_ERR, "FORCE_NOT_EQ failes. Info: %s", __VA_ARGS__);    \
    }                                                                          \
  } while (0)

/*******************************************************************************
 *
 * Global config
 */

/*
  Used by all log related functions.
*/
#define CTH_LOG_ERR stderr
#define CTH_LOG_INFO stdout

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
