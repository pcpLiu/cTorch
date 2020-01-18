#include "common.h"
#include "list.h"
#include "gtest/gtest.h"
#include <stdlib.h>

typedef ListStruct(int) ListTypeName(int);

impl_insert_func(int, ListTypeName(int), ListInsertFuncName(int));
impl_create_func(int, ListTypeName(int), ListItemCreateFuncName(int));

TEST(cTorchListTest, test_create) {
  int x = 3;
  ListTypeName(int) *item = ListItemCreateFuncName(int)(&x);
  EXPECT_EQ(*item->data, x);
  EXPECT_EQ(item->next_item, nullptr);
  EXPECT_EQ(item->prev_item, nullptr);
}

TEST(cTorchListTest, test_insert) {
  int x[] = {1, 2, 3, 4};
  ListTypeName(int) *int_list = ListItemCreateFuncName(int)(&x[0]);
  ListInsertFuncName(int)(int_list, &x[1]);
  ListInsertFuncName(int)(int_list, &x[2]);
  ListInsertFuncName(int)(int_list, &x[3]);

  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(x[i], *int_list->data);
    int_list = int_list->next_item;
  }
}