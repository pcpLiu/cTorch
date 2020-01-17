#ifndef CTH_LIST_H
#define CTH_LIST_H

#include "common.h"

#define ListStruct(data_type)                                                  \
  struct List##data_type {                                                     \
    data_type *data;                                                           \
    struct List##data_type *prev_item;                                         \
    struct List##data_type *next_item;                                         \
  }

#define ListTypeName(data_type) List##data_type

#define declare_insert_func(list_struct, data_type)                            \
  list_struct *insert_into_list_##list_struct(list_struct *, data_type *)

#define impl_insert_func(list_struct, data_type)                               \
  list_struct *insert_into_list_##list_struct(list_struct *head,               \
                                              data_type *data) {               \
    if (head == NULL) {                                                        \
      /* Head pointer is null */                                               \
      head = MALLOC(sizeof(list_struct));                                      \
      head->data = data;                                                       \
    } else if (head->data == NULL) {                                           \
      /* Head data is null */                                                  \
      head->data = data;                                                       \
      return head;                                                             \
    } else {                                                                   \
      list_struct *curr = head;                                                \
      while (curr->next_item != NULL) {                                        \
        curr = curr->next_item;                                                \
      }                                                                        \
      curr->next_item = MALLOC(sizeof(list_struct));                           \
      curr->next_item->data = data;                                            \
      curr->next_item->prev_item = curr;                                       \
    }                                                                          \
    return head;                                                               \
  }

#endif /* LIST_H */
