// Copyright 2021 Zhonghao Liu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CTH_LOGGER_UTIL_H
#define CTH_LOGGER_UTIL_H

#include <stdbool.h>
#include <stdio.h>

/*******************************************************************************
 *
 * Util macros & functions
 */

// TODO: migrate _FAIL_NULL_PTR from a MACRO to a function
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

/**
 * Force a == b. This assumes a, b are primiteve types.
 */
#define FORCE_EQ(a, b, ...)                                                    \
  do {                                                                         \
    if (a != b) {                                                              \
      FAIL_EXIT(CTH_LOG_ERR, __VA_ARGS__);                                     \
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

/**
 * @brief Force expression is true.
 *
 * @note We didn't free `tmp`, `tmp2`, `tmp3` cause system already exits.
 *
 */
#define FORCE_TRUE(expression, ...)                                            \
  do {                                                                         \
    if (!(expression)) {                                                       \
      char *tmp = NULL;                                                        \
      char *tmp2 = NULL;                                                       \
      asprintf(&tmp, "FORCE_TRUE failes: ");                                   \
      asprintf(&tmp2, __VA_ARGS__);                                            \
      char *tmp3 = MALLOC(sizeof(char) * (2 + strlen(tmp) + strlen(tmp2)));    \
      strcpy(tmp3, tmp2);                                                      \
      FAIL_EXIT(CTH_LOG_ERR, tmp3);                                            \
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
  If true, LOG_F() will print out. Otherwise, ommit message.

  Default: true
*/
extern bool CTH_LOG_ENABLE;

#endif /* CTH_LOGGER_UTIL_H */
