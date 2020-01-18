#ifndef CTH_LIST_H
#define CTH_LIST_H

#include "common.h"
#include <stdio.h>
#include <stdlib.h>

#define ListTypeName(data_type) List##data_type
#define ListInsertFuncName(data_type) insert_into_list_List##data_type
#define ListItemInitFuncName(data_type) init_item_List##data_type
#define ListItemCreateFuncName(data_type) create_item_List##data_type
#define ListSizeFuncName(data_type) size_of_List##data_type

#define NewList(list_type) (list_type *)MALLOC(sizeof(list_type))

#define ListStruct(data_type)                                                  \
  struct ListTypeName(data_type) {                                             \
    data_type *data;                                                           \
    struct ListTypeName(data_type) * prev_item;                                \
    struct ListTypeName(data_type) * next_item;                                \
  }

#define declare_init_func(data_type, list_type, func_name)                     \
  list_type *func_name(data_type const *, list_type const *)

#define impl_init_func(data_type, list_type, func_name)                        \
  list_type *func_name(data_type *const data, list_type *const item) {         \
    item->data = data;                                                         \
    item->next_item = NULL;                                                    \
    item->prev_item = NULL;                                                    \
    return item;                                                               \
  }

#define declare_create_func(data_type, list_type, func_name)                   \
  list_type *func_name(data_type *const)

#define impl_create_func(data_type, list_type, func_name)                      \
  list_type *func_name(data_type *const data) {                                \
    list_type *item = (list_type *)MALLOC(sizeof(list_type));                  \
    item->data = data;                                                         \
    item->next_item = NULL;                                                    \
    item->prev_item = NULL;                                                    \
    return item;                                                               \
  }

#define declare_insert_func(data_type, list_type, func_name)                   \
  list_type *func_name(list_type *, data_type *)

#define impl_insert_func(data_type, list_type, func_name)                      \
  list_type *func_name(list_type *head, data_type *data) {                     \
    if (head == NULL) {                                                        \
      /* Head pointer is null */                                               \
      head = (list_type *)MALLOC(sizeof(list_type));                           \
      head->data = data;                                                       \
      head->next_item = NULL;                                                  \
      head->prev_item = NULL;                                                  \
    } else if (head->data == NULL) {                                           \
      /* Head data is null, error */                                           \
      fprintf(stderr, "[cTorch] Item data is NULL.");                          \
      exit(1);                                                                 \
    } else {                                                                   \
      list_type *curr = head;                                                  \
      while (curr->next_item != NULL) {                                        \
        curr = curr->next_item;                                                \
      }                                                                        \
      curr->next_item = (list_type *)MALLOC(sizeof(list_type));                \
      curr->next_item->data = data;                                            \
      curr->next_item->prev_item = curr;                                       \
      curr->next_item->next_item = NULL;                                       \
    }                                                                          \
    return head;                                                               \
  }

#define declare_size_func(list_type, func_name) uint64_t func_name(list_type *)

#define impl_size_func(list_type, func_name)                                   \
  uint64_t func_name(list_type *head) {                                        \
    uint64_t size = 0;                                                         \
    if (head == NULL) {                                                        \
      return size;                                                             \
    } else if (head->data == NULL) {                                           \
      /* TODO: stacktrace */                                                   \
      fprintf(stderr, "[cTorch] Item data is NULL.");                          \
      exit(1);                                                                 \
    } else {                                                                   \
      do {                                                                     \
        size++;                                                                \
        head = head->next_item;                                                \
      } while (head != NULL);                                                  \
    }                                                                          \
    return size;                                                               \
  }

#endif /* LIST_H */
