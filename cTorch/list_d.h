#ifndef CTH_LIST_D_H
#define CTH_LIST_D_H

#include "cTorch/consts.h"
#include "cTorch/logger_util.h"
#include "cTorch/mem_util.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * This file supports a generic Double-linked list structure and supporting
 * functions.
 *
 * This structure is supposed to be used inside library. Some implementatons are
 * based on this assumption.
 *
 * ~~~Structures~~~
 *
 * CTHListItem(T)
 *    - data: T*, contained data
 *    - prev_item: CTHListItem(T)*, previous item
 *    - next_item: CTHListItem(T)*, next item
 *
 * CTHList(T)
 *    - size: list_index_t, size of list
 *    - head: CTHListItem(T)*, head item of list
 *    - tail: CTHListItem(T)*, tail item of list
 *
 * ~~~Functions~~~
 *
 * CTHListItem(T)* cth_new_list_item(T)(T* data)
 *    - Create a new list item. Caller needs tacking care of its memory
 * releasing
 *
 * CTHList(T)* cth_new_list(T)()
 *    - Create a new list, caller needs taking care of its memory releasing
 *
 * void cth_insert_list(T)(CTHList(T)* list, T* data)
 *    - Insert a data into a list
 *
 * bool cth_list_contains_data(T)(CTHList(T)* list, T* data)
 *    - Check if list contains a item has same data value as given data
 *
 * T* cth_list_pop(T)(CTHList(T)* list)
 *    - Pop the head item of the list
 *    - Function will free popped data's item
 *
 * T* cth_list_at(T)(CTHList(T) *list, list_index_t index)
 *    - Get data at given index
 *
 * void cth_free_list(T)(CTHList(T) *list)
 *    - Free a list and all items it contains. This function does not free data
 *
 * void cth_free_list_deep(T)(CTHList(T) *list)
 *    - Free a list and all items it contains. Also, it frees stored data.
 *    - We assume contained data has a corresponding deep free function
 *
 * void cth_list_del(T)(CTHList(T) *list, T *data)
 *    - Delete an item from list that contains given data
 *    - The item itself will not be free
 */

/**
 * List index type
 */
typedef uint32_t list_index_t;

// Generic list item struct
//
// CTHListItem(data_type) --- list item type name
// CTHListItemStruct(data_type) --- list item struct
// cth_def_list_item(data_type) --- Define new list item type
#define CTHListItem(data_type) ListItem##data_type
#define CTHListItemStruct(data_type)                                           \
  struct CTHListItem(data_type) {                                              \
    data_type *data;                                                           \
    struct CTHListItem(data_type) * prev_item;                                 \
    struct CTHListItem(data_type) * next_item;                                 \
  }
#define cth_def_list_item(data_type)                                           \
  typedef CTHListItemStruct(data_type) CTHListItem(data_type)

// Create a non-connected item with given data.
// Function fails on empty.
//
// cth_new_list_item(data_type) --- func name
// cth_declare_new_list_item_func(data_type) --- declaration
// cth_impl_new_list_item_func(data_type) --- implementation
#define cth_new_list_item(data_type) new_list_item##data_type
#define _cth_declare_new_list_item_func(data_type, func_name)                  \
  CTHListItem(data_type) * func_name(data_type *)
#define cth_declare_new_list_item_func(data_type)                              \
  _cth_declare_new_list_item_func(data_type, cth_new_list_item(data_type))
#define _cth_impl_new_list_item_func(data_type, list_type, func_name)          \
  list_type *func_name(data_type *data) {                                      \
    FAIL_NULL_PTR(data);                                                       \
    list_type *item = (list_type *)MALLOC(sizeof(list_type));                  \
    item->data = data;                                                         \
    item->prev_item = NULL;                                                    \
    item->next_item = NULL;                                                    \
    return item;                                                               \
  }
#define cth_impl_new_list_item_func(data_type)                                 \
  _cth_impl_new_list_item_func(                                                \
      data_type, CTHListItem(data_type), cth_new_list_item(data_type))

// Generic double-linked list struct
//
// CTHList(data_type) --- list type name
// CTHListStruct(data_type) --- list struct
// def_list(data_type) --- define list type
#define CTHList(data_type) List##data_type
#define CTHListStruct(data_type)                                               \
  struct CTHList(data_type) {                                                  \
    list_index_t size;                                                         \
    struct CTHListItem(data_type) * head;                                      \
    struct CTHListItem(data_type) * tail;                                      \
  }
#define def_list(data_type) typedef CTHListStruct(data_type) CTHList(data_type)

// Create an empty list.
//
// cth_new_list(data_type) --- func name
// cth_declare_new_list_func(data_type) --- declaration
// cth_impl_new_list_func(data_type) --- implementation
#define cth_new_list(data_type) new_list##data_type
#define _cth_declare_new_list_func(list_type, func_name) list_type *func_name()
#define cth_declare_new_list_func(data_type)                                   \
  _cth_declare_new_list_func(CTHList(data_type), cth_new_list(data_type))
#define _cth_impl_new_list_func(list_type, func_name)                          \
  list_type *func_name() {                                                     \
    list_type *list = (list_type *)MALLOC(sizeof(list_type));                  \
    list->head = NULL;                                                         \
    list->tail = NULL;                                                         \
    list->size = 0;                                                            \
    return list;                                                               \
  }
#define cth_impl_new_list_func(data_type)                                      \
  _cth_impl_new_list_func(CTHList(data_type), cth_new_list(data_type))

// Insert data into a list. Fails on empty inputs.
// Funtion returns created item for given data.
//
// cth_insert_list(data_type) --- func name
// cth_declare_insert_list_func(data_type) --- declaration
// cth_impl_insert_list_func(data_type) --- implementation
#define cth_insert_list(data_type) insert_list_##data_type
#define _cth_declare_insert_list_func(                                         \
    data_type, item_type, list_type, func_name)                                \
  item_type *func_name(list_type *, data_type *)
#define cth_declare_insert_list_func(data_type)                                \
  _cth_declare_insert_list_func(                                               \
      data_type,                                                               \
      CTHListItem(data_type),                                                  \
      CTHList(data_type),                                                      \
      cth_insert_list(data_type))
#define _cth_impl_insert_list_func(data_type, item_type, list_type, func_name) \
  item_type *func_name(list_type *list, data_type *data) {                     \
    FAIL_NULL_PTR(list);                                                       \
    FAIL_NULL_PTR(data);                                                       \
    CTHListItem(data_type) *item = cth_new_list_item(data_type)(data);         \
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
#define cth_impl_insert_list_func(data_type)                                   \
  _cth_impl_insert_list_func(                                                  \
      data_type,                                                               \
      CTHListItem(data_type),                                                  \
      CTHList(data_type),                                                      \
      cth_insert_list(data_type))

// Check if list contains a data (by address). Fails on empty inputs.
// Function returns item if found. Else, returns NULL.
// It returns NULL if list is empty.
//
// cth_list_contains_data(data_type) --- func name
// cth_declare_list_contains_data_func(data_type) --- declaration
// cth_impl_list_contains_data_func(data_type) --- implementation
#define cth_list_contains_data(data_type) list_contains_data_##data_type
#define _cth_declare_list_contains_data_func(                                  \
    data_type, item_type, list_type, func_name)                                \
  item_type *func_name(list_type *, data_type *)
#define cth_declare_list_contains_data_func(data_type)                         \
  _cth_declare_list_contains_data_func(                                        \
      data_type,                                                               \
      CTHListItem(data_type),                                                  \
      CTHList(data_type),                                                      \
      cth_list_contains_data(data_type))
#define _cth_impl_list_contains_data_func(                                     \
    data_type, item_type, list_type, func_name)                                \
  item_type *func_name(list_type *list, data_type *data) {                     \
    FAIL_NULL_PTR(list);                                                       \
    FAIL_NULL_PTR(data);                                                       \
    item_type *found = NULL;                                                   \
    if (list->size != 0) {                                                     \
      CTHListItem(data_type) *item = list->head;                               \
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
#define cth_impl_list_contains_data_func(data_type)                            \
  _cth_impl_list_contains_data_func(                                           \
      data_type,                                                               \
      CTHListItem(data_type),                                                  \
      CTHList(data_type),                                                      \
      cth_list_contains_data(data_type))

// Check if list contains a item (by address). Fails on empty inputs.
// Function returns false if list is empty.
//
// cth_list_contains_item(data_type) --- func name
// cth_declare_list_contains_item_func(data_type) --- declaration
// cth_impl_list_contains_item_func(data_type) --- implementation
#define cth_list_contains_item(data_type) list_contains_item_##data_type
#define _cth_declare_list_contains_item_func(item_type, list_type, func_name)  \
  bool func_name(list_type *, item_type *)
#define cth_declare_list_contains_item_func(data_type)                         \
  _cth_declare_list_contains_item_func(                                        \
      CTHListItem(data_type),                                                  \
      CTHList(data_type),                                                      \
      cth_list_contains_item(data_type))
#define _cth_impl_list_contains_item_func(item_type, list_type, func_name)     \
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
#define cth_impl_list_contains_item_func(data_type)                            \
  _cth_impl_list_contains_item_func(                                           \
      CTHListItem(data_type),                                                  \
      CTHList(data_type),                                                      \
      cth_list_contains_item(data_type))

// Pop head item's data of a list. If list is empty, return NULL.
//
// cth_list_pop(data_type) --- func name
// cth_declare_list_pop_func(data_type) --- declare func
// cth_impl_list_pop_func(data_type) --- impl func
#define cth_list_pop(data_type) list_pop_##data_type
#define _cth_declare_list_pop_func(data_type, list_type, func_name)            \
  data_type *func_name(list_type *)
#define cth_declare_list_pop_func(data_type)                                   \
  _cth_declare_list_pop_func(                                                  \
      data_type, CTHList(data_type), cth_list_pop(data_type))
#define _cth_impl_list_pop_func(data_type, list_type, func_name)               \
  data_type *func_name(list_type *list) {                                      \
    FAIL_NULL_PTR(list);                                                       \
    if (list->size == 0) {                                                     \
      return NULL;                                                             \
    }                                                                          \
                                                                               \
    CTHListItem(data_type) *old_head = list->head;                             \
    list->head = old_head->next_item;                                          \
    if (list->head && list->head->prev_item != NULL) {                         \
      list->head->prev_item = NULL;                                            \
    }                                                                          \
    list->size--;                                                              \
    if (list->size == 0) {                                                     \
      list->tail = NULL;                                                       \
    }                                                                          \
                                                                               \
    data_type *ret = old_head->data;                                           \
    FREE(old_head);                                                            \
    return ret;                                                                \
  }
#define cth_impl_list_pop_func(data_type)                                      \
  _cth_impl_list_pop_func(                                                     \
      data_type, CTHList(data_type), cth_list_pop(data_type))

// Get item's data given a 0-based index. If index > list size, error happened.
//
// cth_list_at(data_type) --- func name
// cth_declare_list_at_func(data_type) --- declare func
// cth_impl_list_at_func(data_type) --- impl func
#define cth_list_at(data_type) list_at_##data_type
#define _cth_declare_list_at_func(data_type, list_type, func_name)             \
  data_type *func_name(list_type *list, list_index_t index)
#define cth_declare_list_at_func(data_type)                                    \
  _cth_declare_list_at_func(                                                   \
      data_type, CTHList(data_type), cth_list_at(data_type))
#define _cth_impl_list_at_func(data_type, list_type, func_name)                \
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
    CTHListItem(data_type) *item = NULL;                                       \
    if (index < list->size / 2) {                                              \
      item = list->head;                                                       \
      list_index_t i = 0;                                                      \
      while (i != index) {                                                     \
        item = item->next_item;                                                \
        i++;                                                                   \
      };                                                                       \
    } else {                                                                   \
      item = list->tail;                                                       \
      list_index_t i = list->size - 1;                                         \
      while (i != index) {                                                     \
        item = item->prev_item;                                                \
        i--;                                                                   \
      };                                                                       \
    }                                                                          \
    return item->data;                                                         \
  }
#define cth_impl_list_at_func(data_type)                                       \
  _cth_impl_list_at_func(data_type, CTHList(data_type), cth_list_at(data_type))

// Free a list and all its items. It would not free data.
//
// cth_free_list(data_type) --- func name
// cth_declare_free_list_func(data_type) --- declaration
//
#define cth_free_list(data_type) free_list_##data_type
#define _cth_declare_free_list_func(data_type, list_type, func_name)           \
  void func_name(list_type *list)
#define cth_declare_free_list_func(data_type)                                  \
  _cth_declare_free_list_func(                                                 \
      data_type, CTHList(data_type), cth_free_list(data_type))
#define _cth_impl_free_list_func(data_type, list_type, func_name)              \
  void func_name(list_type *list) {                                            \
    FAIL_NULL_PTR(list);                                                       \
    while (list->size > 0) {                                                   \
      cth_list_pop(data_type)(list);                                           \
    }                                                                          \
    FREE(list);                                                                \
  }
#define cth_impl_free_list_func(data_type)                                     \
  _cth_impl_free_list_func(                                                    \
      data_type, CTHList(data_type), cth_free_list(data_type))

// Free a list and all its items. Also it frees all data.
//
// cth_free_list(data_type) --- func name
// cth_declare_free_list_func(data_type) --- declaration
//
#define cth_free_list_deep(data_type) free_list_deep_##data_type
#define _cth_declare_free_list_deep_func(data_type, list_type, func_name)      \
  void func_name(list_type *list)
#define cth_declare_free_list_deep_func(data_type)                             \
  _cth_declare_free_list_deep_func(                                            \
      data_type, CTHList(data_type), cth_free_list_deep(data_type))
#define _cth_impl_free_list_deep_func(                                         \
    data_type, list_type, data_free_func, func_name)                           \
  void func_name(list_type *list) {                                            \
    FAIL_NULL_PTR(list);                                                       \
    while (list->size > 0) {                                                   \
      data_type *data = cth_list_pop(data_type)(list);                         \
      data_free_func(data);                                                    \
    }                                                                          \
    FREE(list);                                                                \
  }
#define cth_impl_free_list_deep_func(data_type)                                \
  _cth_impl_free_list_deep_func(                                               \
      data_type,                                                               \
      CTHList(data_type),                                                      \
      struct_deep_free(data_type),                                             \
      cth_free_list_deep(data_type))

// Delete an item from a list
//
// cth_list_del(data_type) --- func name
// cth_declare_list_del_func(data_type) --- declaration
// cth_impl_list_del_func(data_type) --- implementation
//
#define cth_list_del(data_type) list_del_##data_type
#define _cth_declare_list_del_func(data_type, list_type, func_name)            \
  void func_name(list_type *list, const data_type *data)
#define cth_declare_list_del_func(data_type)                                   \
  _cth_declare_list_del_func(                                                  \
      data_type, CTHList(data_type), cth_list_del(data_type))
#define _cth_impl_list_del_func(data_type, list_type, func_name)               \
  void func_name(list_type *list, const data_type *target_data) {              \
    FAIL_NULL_PTR(list);                                                       \
    FAIL_NULL_PTR(target_data);                                                \
    CTHListItem(data_type) *item = list->head;                                 \
    while (item != NULL) {                                                     \
      if (target_data == item->data) {                                         \
        if (item->prev_item != NULL) {                                         \
          item->prev_item->next_item = item->next_item;                        \
        }                                                                      \
        if (item->next_item != NULL) {                                         \
          item->next_item->prev_item = item->prev_item;                        \
        }                                                                      \
        list->size--;                                                          \
                                                                               \
        if (item == list->head) {                                              \
          list->head = item->next_item;                                        \
        }                                                                      \
        if (item == list->tail) {                                              \
          list->tail = item->prev_item;                                        \
        }                                                                      \
                                                                               \
        FREE(item);                                                            \
        break;                                                                 \
      }                                                                        \
      item = item->next_item;                                                  \
    }                                                                          \
  }
#define cth_impl_list_del_func(data_type)                                      \
  _cth_impl_list_del_func(                                                     \
      data_type, CTHList(data_type), cth_list_del(data_type))

#endif /* LIST_D_H */
