#ifndef CTH_LIST_D_H
#define CTH_LIST_D_H

#include "cTorch/common.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * This file supports a generic Double-linked list structure and supporting
 * functions.
 *
 *
 * ~~~Structures~~~
 *
 * ListItem(T)
 *  - data: T*, contained data
 *  - prev_item: ListItem(T)*, previous item
 *  - next_item: ListItem(T)*, next item
 *
 * List(T)
 *  - size: list_index_t, size of list
 *  - head: ListItem(T)*, head item of list
 *  - tail: ListItem(T)*, tail item of list
 *
 * ~~~Functions~~~
 *
 * ListItem(T)* new_list_item(T)(T* data)
 *  - Create a new list item. Caller needs tacking care of its memory releasing
 *
 * List(T)* new_list(T)()
 *  - Create a new list, caller needs taking care of its memory releasing
 *
 * void insert_list(T)(List(T)* list, T* data)
 *  - Insert a data into a list
 *
 * bool list_contains_data(T)(List(T)* list, T* data)
 *  - Check if list contains a item has same data value as given data
 *
 * T* list_pop(T)(List(T)* list)
 *  - Pop the head item of the list
 *  - Function will free popped data's item
 *
 * T* list_at(T)(List(T) *list, list_index_t index)
 *  - Get data at given index
 *
 * void free_list(T)(List(T) *list)
 *  - Free a list and all items it contains. This function does not free data.
 */

/*
  List index type
*/
typedef int32_t list_index_t;

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
// impl_new_list_item_func(data_type) --- implementation
#define new_list_item(data_type) new_list_item##data_type
#define _declare_new_list_item_func(data_type, func_name)                      \
  ListItem(data_type) * func_name(data_type *)
#define declare_new_list_item_func(data_type)                                  \
  _declare_new_list_item_func(data_type, new_list_item(data_type))
#define _impl_new_list_item_func(data_type, list_type, func_name)              \
  list_type *func_name(data_type *data) {                                      \
    FAIL_NULL_PTR(data);                                                       \
    list_type *item = (list_type *)MALLOC(sizeof(list_type));                  \
    item->data = data;                                                         \
    item->prev_item = NULL;                                                    \
    item->next_item = NULL;                                                    \
    return item;                                                               \
  }
#define impl_new_list_item_func(data_type)                                     \
  _impl_new_list_item_func(                                                    \
      data_type,                                                               \
      ListItem(data_type),                                                     \
      new_list_item(data_type))

// Generic double-linked list struct
//
// List(data_type) --- list type name
// ListStruct(data_type) --- list struct
// def_list(data_type) --- define list type
#define List(data_type) List##data_type
#define ListStruct(data_type)                                                  \
  struct List(data_type) {                                                     \
    list_index_t size;                                                         \
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
  _declare_insert_list_func(                                                   \
      data_type,                                                               \
      ListItem(data_type),                                                     \
      List(data_type),                                                         \
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
  _impl_insert_list_func(                                                      \
      data_type,                                                               \
      ListItem(data_type),                                                     \
      List(data_type),                                                         \
      insert_list(data_type))

// Check if list contains a data (by address). Fails on empty inputs.
// Function returns item if found. Else, returns NULL.
// It returns NULL if list is empty.
//
// list_contains_data(data_type) --- func name
// declare_list_contains_data_func(data_type) --- declaration
// impl_list_contains_data_func(data_type) --- implementation
#define list_contains_data(data_type) list_contains_data_##data_type
#define _declare_list_contains_data_func(                                      \
    data_type,                                                                 \
    item_type,                                                                 \
    list_type,                                                                 \
    func_name)                                                                 \
  item_type *func_name(list_type *, data_type *)
#define declare_list_contains_data_func(data_type)                             \
  _declare_list_contains_data_func(                                            \
      data_type,                                                               \
      ListItem(data_type),                                                     \
      List(data_type),                                                         \
      list_contains_data(data_type))
#define _impl_list_contains_data_func(                                         \
    data_type,                                                                 \
    item_type,                                                                 \
    list_type,                                                                 \
    func_name)                                                                 \
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
  _impl_list_contains_data_func(                                               \
      data_type,                                                               \
      ListItem(data_type),                                                     \
      List(data_type),                                                         \
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
  _declare_list_contains_item_func(                                            \
      ListItem(data_type),                                                     \
      List(data_type),                                                         \
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
  _impl_list_contains_item_func(                                               \
      ListItem(data_type),                                                     \
      List(data_type),                                                         \
      list_contains_item(data_type))

// Pop head item's data of a list. If list is empty, return NULL.
//
// list_pop(data_type) --- func name
// declare_list_pop_func(data_type) --- declare func
// impl_list_pop_func(data_type) --- impl func
#define list_pop(data_type) list_pop_##data_type
#define _declare_list_pop_func(data_type, list_type, func_name)                \
  data_type *func_name(list_type *)
#define declare_list_pop_func(data_type)                                       \
  _declare_list_pop_func(data_type, List(data_type), list_pop(data_type))
#define _impl_list_pop_func(data_type, list_type, func_name)                   \
  data_type *func_name(list_type *list) {                                      \
    FAIL_NULL_PTR(list);                                                       \
    if (list->size == 0) {                                                     \
      return NULL;                                                             \
    }                                                                          \
                                                                               \
    ListItem(data_type) *old_head = list->head;                                \
    list->head = old_head->next_item;                                          \
    list->size--;                                                              \
    if (list->size == 0) {                                                     \
      list->tail = NULL;                                                       \
    }                                                                          \
                                                                               \
    data_type *ret = old_head->data;                                           \
    FREE((void **)&old_head);                                                  \
    return ret;                                                                \
  }
#define impl_list_pop_func(data_type)                                          \
  _impl_list_pop_func(data_type, List(data_type), list_pop(data_type))

// Get item's data given a 0-based index. If index > list size, error happened.
//
// list_at(data_type) --- func name
// declare_list_at_func(data_type) --- declare func
// impl_list_at_func(data_type) --- impl func
#define list_at(data_type) list_at_##data_type
#define _declare_list_at_func(data_type, list_type, func_name)                 \
  data_type *func_name(list_type *list, list_index_t index)
#define declare_list_at_func(data_type)                                        \
  _declare_list_at_func(data_type, List(data_type), list_at(data_type))
#define _impl_list_at_func(data_type, list_type, func_name)                    \
  data_type *func_name(list_type *list, list_index_t index) {                  \
    FAIL_NULL_PTR(list);                                                       \
    if (index >= list->size) {                                                 \
      FAIL_EXIT(                                                               \
          CTH_LOG_ERR,                                                         \
          "Error at func list_at: Given index %d is larger than or "           \
          "equal to list size %d.",                                            \
          index,                                                               \
          list->size);                                                         \
    }                                                                          \
                                                                               \
    ListItem(data_type) *item = list->head;                                    \
    list_index_t i = 0;                                                        \
    while (i != index) {                                                       \
      item = item->next_item;                                                  \
      i++;                                                                     \
    };                                                                         \
    return item->data;                                                         \
  }
#define impl_list_at_func(data_type)                                           \
  _impl_list_at_func(data_type, List(data_type), list_at(data_type))

// Free a list and all its items. It would not free data.
//
// free_list(data_type) --- func name
// declare_free_list_func(data_type) --- declaration
//
#define free_list(data_type) free_list_##data_type
#define _declare_free_list_func(data_type, list_type, func_name)               \
  void func_name(list_type *list)
#define declare_free_list_func(data_type)                                      \
  _declare_free_list_func(data_type, List(data_type), free_list(data_type))
#define _impl_free_list_func(data_type, list_type, func_name)                  \
  void func_name(list_type *list) {                                            \
    FAIL_NULL_PTR(list);                                                       \
    while (list->size > 0) {                                                   \
      list_pop(data_type)(list);                                               \
    }                                                                          \
    FREE((void **)&list);                                                      \
  }
#define impl_free_list_func(data_type)                                         \
  _impl_free_list_func(data_type, List(data_type), free_list(data_type))

#endif /* LIST_D_H */
