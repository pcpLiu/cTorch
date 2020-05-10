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
void FAIL_NULL_PTR(void *);

/*
  Print error message and exit.
*/
#define FAIL_EXIT(write_to, ...)                                               \
  do {                                                                         \
    char *tmp = NULL;                                                          \
    fprintf(stderr, "[cTorch]: ");                                             \
    asprintf(&tmp, __VA_ARGS__);                                               \
    fprintf(stderr, "%s\n", tmp);                                              \
    FREE((void **)&tmp);                                                       \
    exit(1);                                                                   \
  } while (0)

/*
  Log message. Auto break line at the end.

  Will check global config
*/
#define LOG_F() F

/*
  Exit if val is NAN.

  If global config CTH_NAN_EXIT is false, it will not exit.
*/
#define NAN_EXIT(val)                                                          \
  do {                                                                         \
    if (val != val && CTH_NAN_EXIT == true) {                                  \
      FAIL_EXIT(CTH_LOG_STR, "Value is NaN");                                  \
    }                                                                          \
  } while (0)

/*******************************************************************************
 *
 * Global config
 */

/*
  Used by all log related functions.
*/
extern char *CTH_LOG_STR;

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
