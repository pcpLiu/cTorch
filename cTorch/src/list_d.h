#ifndef CTH_LIST_D_H
#define CTH_LIST_D_H

#include "common.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Generic list item struct
//
// ListItem(data_type) --- list item type name
// ListItemStruct(data_type) --- list item struct
// def_list_item(data_type) --- Define new list item type
#define ListItem(data_type) ListItem##data_type
#define ListItemStruct(data_type)                                              \
  struct ListItem(data_type) {                                                 \
    data_type *data;                                                           \
    struct ListItem(data_type) * prev_item;                                    \
    struct ListItem(data_type) * next_item;                                    \
  }
#define def_list_item(data_type)                                               \
  typedef ListItemStruct(data_type) ListItem(data_type)

// Create a non-connected item with given data.
// Function fails on empty.
//
// new_list_item(data_type) --- func name
// declare_new_list_item_func(data_type) --- declaration
// impl_new_new_list_item_func(data_type) --- implementation
#define new_list_item(data_type) new_list_item##data_type
#define _declare_new_list_item_func(data_type, func_name)                      \
  ListItem(data_type) * func_name(data_type *)
#define declare_new_list_item_func(data_type)                                  \
  _declare_new_list_item_func(data_type, new_list_item(data_type))
#define _impl_new_new_list_item_func(data_type, list_type, func_name)          \
  list_type *func_name(data_type *data) {                                      \
    FAIL_NULL_PTR(data);                                                       \
    list_type *item = (list_type *)MALLOC(sizeof(list_type));                  \
    item->data = data;                                                         \
    item->prev_item = NULL;                                                    \
    item->next_item = NULL;                                                    \
    return item;                                                               \
  }
#define impl_new_new_list_item_func(data_type)                                 \
  _impl_new_new_list_item_func(data_type, ListItem(data_type),                 \
                               new_list_item(data_type))

// Generic double-linked list struct
//
// List(data_type) --- list type name
// ListStruct(data_type) --- list struct
// def_list(data_type) --- define list type
#define List(data_type) List##data_type
#define ListStruct(data_type)                                                  \
  struct List(data_type) {                                                     \
    uint32_t size;                                                             \
    struct ListItem(data_type) * head;                                         \
    struct ListItem(data_type) * tail;                                         \
  }
#define def_list(data_type) typedef ListStruct(data_type) List(data_type)

// Create an empty list.
//
// new_list(data_type) --- func name
// declare_new_list_func(data_type) --- declaration
// impl_new_list_func(data_type) --- implementation
#define new_list(data_type) new_list##data_type
#define _declare_new_list_func(list_type, func_name) list_type *func_name()
#define declare_new_list_func(data_type)                                       \
  _declare_new_list_func(List(data_type), new_list(data_type))
#define _impl_new_list_func(list_type, func_name)                              \
  list_type *func_name() {                                                     \
    list_type *list = (list_type *)MALLOC(sizeof(list_type));                  \
    list->head = NULL;                                                         \
    list->tail = NULL;                                                         \
    list->size = 0;                                                            \
    return list;                                                               \
  }
#define impl_new_list_func(data_type)                                          \
  _impl_new_list_func(List(data_type), new_list(data_type))

// Insert data into a list. Fails on empty inputs.
// Funtion returns created item for given data.
//
// insert_list(data_type) --- func name
// declare_insert_list_func(data_type) --- declaration
// impl_insert_list_func(data_type) --- implementation
#define insert_list(data_type) insert_list_##data_type
#define _declare_insert_list_func(data_type, item_type, list_type, func_name)  \
  item_type *func_name(list_type *, data_type *)
#define declare_insert_list_func(data_type)                                    \
  _declare_insert_list_func(data_type, ListItem(data_type), List(data_type),   \
                            insert_list(data_type))
#define _impl_insert_list_func(data_type, item_type, list_type, func_name)     \
  item_type *func_name(list_type *list, data_type *data) {                     \
    FAIL_NULL_PTR(list);                                                       \
    FAIL_NULL_PTR(data);                                                       \
    ListItem(data_type) *item = new_list_item(data_type)(data);                \
    if (list->size == 0) {                                                     \
      list->tail = item;                                                       \
      list->head = item;                                                       \
    } else {                                                                   \
      list->tail->next_item = item;                                            \
      item->prev_item = list->tail;                                            \
      list->tail = item;                                                       \
    }                                                                          \
    list->size = list->size + 1;                                               \
    return item;                                                               \
  }
#define impl_insert_list_func(data_type)                                       \
  _impl_insert_list_func(data_type, ListItem(data_type), List(data_type),      \
                         insert_list(data_type))

// Check if list contains a data (by address). Fails on empty inputs.
// Function returns item if found. Else, returns NULL.
// It returns NULL if list is empty.
//
// list_contains_data(data_type) --- func name
// declare_list_contains_data_func(data_type) --- declaration
// impl_list_contains_data_func(data_type) --- implementation
#define list_contains_data(data_type) list_contains_data_##data_type
#define _declare_list_contains_data_func(data_type, item_type, list_type,      \
                                         func_name)                            \
  item_type *func_name(list_type *, data_type *)
#define declare_list_contains_data_func(data_type)                             \
  _declare_list_contains_data_func(data_type, ListItem(data_type),             \
                                   List(data_type),                            \
                                   list_contains_data(data_type))
#define _impl_list_contains_data_func(data_type, item_type, list_type,         \
                                      func_name)                               \
  item_type *func_name(list_type *list, data_type *data) {                     \
    FAIL_NULL_PTR(list);                                                       \
    FAIL_NULL_PTR(data);                                                       \
    item_type *found = NULL;                                                   \
    if (list->size != 0) {                                                     \
      ListItem(data_type) *item = list->head;                                  \
      do {                                                                     \
        if (item->data == data) {                                              \
          found = item;                                                        \
          break;                                                               \
        }                                                                      \
        item = item->next_item;                                                \
      } while (item != NULL);                                                  \
    }                                                                          \
    return found;                                                              \
  }
#define impl_list_contains_data_func(data_type)                                \
  _impl_list_contains_data_func(data_type, ListItem(data_type),                \
                                List(data_type),                               \
                                list_contains_data(data_type))

// Check if list contains a item (by address). Fails on empty inputs.
// Function returns false if list is empty.
//
// list_contains_item(data_type) --- func name
// declare_list_contains_item_func(data_type) --- declaration
// impl_list_contains_item_func(data_type) --- implementation
#define list_contains_item(data_type) list_contains_item_##data_type
#define _declare_list_contains_item_func(item_type, list_type, func_name)      \
  bool func_name(list_type *, item_type *)
#define declare_list_contains_item_func(data_type)                             \
  _declare_list_contains_item_func(ListItem(data_type), List(data_type),       \
                                   list_contains_item(data_type))
#define _impl_list_contains_item_func(item_type, list_type, func_name)         \
  bool func_name(list_type *list, item_type *target_item) {                    \
    FAIL_NULL_PTR(list);                                                       \
    FAIL_NULL_PTR(target_item);                                                \
    bool contain = false;                                                      \
    if (list->size != 0) {                                                     \
      item_type *item = list->head;                                            \
      do {                                                                     \
        if (item == target_item) {                                             \
          contain = true;                                                      \
          break;                                                               \
        }                                                                      \
        item = item->next_item;                                                \
      } while (item != NULL);                                                  \
    }                                                                          \
    return contain;                                                            \
  }
#define impl_list_contains_item_func(data_type)                                \
  _impl_list_contains_item_func(ListItem(data_type), List(data_type),          \
                                list_contains_item(data_type))

#endif /* LIST_D_H */
