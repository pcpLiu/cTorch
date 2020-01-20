#include "c_torch.h"
#include "gtest/gtest.h"
#include <stdlib.h>

// typedef ListItemStruct(int) ListItem(int);
def_list_item(int);
def_list(int);
impl_new_new_list_item_func(int);
impl_new_list_func(int);
impl_insert_list_func(int);
impl_list_contains_data_func(int);
impl_list_contains_item_func(int);

TEST(cTorchListTest, test_item_define) {
  int a = 3;
  ListItem(int) item = {
      .data = &a,
      .prev_item = NULL,
      .next_item = NULL,
  };
  EXPECT_EQ(*item.data, 3);
  EXPECT_EQ(item.prev_item, nullptr);
  EXPECT_EQ(item.next_item, nullptr);
}

TEST(cTorchListTest, test_list_define) {
  List(int) list = {
      .head = NULL,
      .tail = NULL,
      .size = 0,
  };
  EXPECT_EQ(list.size, 0);
  EXPECT_EQ(list.head, nullptr);
  EXPECT_EQ(list.tail, nullptr);
}

TEST(cTorchListTest, test_new_list_item_func) {
  int a = 3;
  ListItem(int) *item = new_list_item(int)(&a);
  EXPECT_EQ(*item->data, 3);
  EXPECT_EQ(item->prev_item, nullptr);
  EXPECT_EQ(item->next_item, nullptr);
}

TEST(cTorchListTest, test_new_list_item_func_fail) {
  EXPECT_EXIT(new_list_item(int)(NULL), ::testing::ExitedWithCode(1),
              "NULL ptr error");
}

TEST(cTorchListTest, test_new_list_func) {
  List(int) *list = new_list(int)();
  EXPECT_EQ(list->size, 0);
  EXPECT_EQ(list->head, nullptr);
  EXPECT_EQ(list->tail, nullptr);
}

TEST(cTorchListTest, test_list_insert_func) {
  int x[] = {1, 2, 3, 4};
  List(int) *list = new_list(int)();
  ListItem(int) *item_1 = insert_list(int)(list, &x[0]);
  ListItem(int) *item_2 = insert_list(int)(list, &x[1]);
  ListItem(int) *item_3 = insert_list(int)(list, &x[2]);
  ListItem(int) *item_4 = insert_list(int)(list, &x[3]);

  EXPECT_EQ(list->size, 4);
  EXPECT_EQ(list->head, item_1);
  EXPECT_EQ(list->tail, item_4);

  EXPECT_EQ(item_1->prev_item, nullptr);
  EXPECT_EQ(item_1->next_item, item_2);
  EXPECT_EQ(*item_1->data, x[0]);

  EXPECT_EQ(item_2->prev_item, item_1);
  EXPECT_EQ(item_2->next_item, item_3);
  EXPECT_EQ(*item_2->data, x[1]);

  EXPECT_EQ(item_3->prev_item, item_2);
  EXPECT_EQ(item_3->next_item, item_4);
  EXPECT_EQ(*item_3->data, x[2]);

  EXPECT_EQ(item_4->prev_item, item_3);
  EXPECT_EQ(item_4->next_item, nullptr);
  EXPECT_EQ(*item_4->data, x[3]);
}

TEST(cTorchListTest, test_list_insert_func_fail) {
  int x[] = {1, 2, 3, 4};
  List(int) *list = new_list(int)();

  EXPECT_EXIT(insert_list(int)(NULL, &x[0]), ::testing::ExitedWithCode(1),
              "NULL ptr error");
  EXPECT_EXIT(insert_list(int)(list, NULL), ::testing::ExitedWithCode(1),
              "NULL ptr error");
  EXPECT_EXIT(insert_list(int)(NULL, NULL), ::testing::ExitedWithCode(1),
              "NULL ptr error");
}

TEST(cTorchListTest, test_list_contains) {
  int x[] = {1, 2, 3, 4, 5};
  List(int) *list = new_list(int)();
  ListItem(int) *item_1 = insert_list(int)(list, &x[0]);
  ListItem(int) *item_2 = insert_list(int)(list, &x[1]);
  ListItem(int) *item_3 = insert_list(int)(list, &x[2]);
  ListItem(int) *item_4 = insert_list(int)(list, &x[3]);

  ListItem(int) * found_item;
  found_item = list_contains_data(int)(list, &x[0]);
  EXPECT_EQ(item_1, found_item);
  found_item = list_contains_data(int)(list, &x[1]);
  EXPECT_EQ(item_2, found_item);
  found_item = list_contains_data(int)(list, &x[2]);
  EXPECT_EQ(item_3, found_item);
  found_item = list_contains_data(int)(list, &x[3]);
  EXPECT_EQ(item_4, found_item);
  found_item = list_contains_data(int)(list, &x[4]);
  EXPECT_EQ(nullptr, found_item);
}

TEST(cTorchListTest, test_list_contains_fails) {
  int x[] = {1, 2, 3, 4};
  List(int) *list = new_list(int)();

  EXPECT_EXIT(list_contains_data(int)(NULL, &x[0]),
              ::testing::ExitedWithCode(1), "NULL ptr error");
  EXPECT_EXIT(list_contains_data(int)(list, NULL), ::testing::ExitedWithCode(1),
              "NULL ptr error");
  EXPECT_EXIT(list_contains_data(int)(NULL, NULL), ::testing::ExitedWithCode(1),
              "NULL ptr error");
}

TEST(cTorchListTest, test_list_contains_item) {
  int x[] = {1, 2, 3, 4, 5};
  List(int) *list = new_list(int)();
  ListItem(int) *item_1 = insert_list(int)(list, &x[0]);
  ListItem(int) *item_2 = insert_list(int)(list, &x[1]);
  ListItem(int) *item_3 = insert_list(int)(list, &x[2]);
  ListItem(int) *item_4 = insert_list(int)(list, &x[3]);
  ListItem(int) *item_5 = new_list_item(int)(&x[4]);

  bool found;
  found = list_contains_item(int)(list, item_1);
  EXPECT_EQ(true, found);
  found = list_contains_item(int)(list, item_2);
  EXPECT_EQ(true, found);
  found = list_contains_item(int)(list, item_3);
  EXPECT_EQ(true, found);
  found = list_contains_item(int)(list, item_4);
  EXPECT_EQ(true, found);
  found = list_contains_item(int)(list, item_5);
  EXPECT_EQ(false, found);
}

TEST(cTorchListTest, test_list_contains_item_fails) {
  int x[] = {1, 2, 3, 4};
  ListItem(int) *item = new_list_item(int)(&x[4]);
  List(int) *list = new_list(int)();

  EXPECT_EXIT(list_contains_item(int)(NULL, item), ::testing::ExitedWithCode(1),
              "NULL ptr error");
  EXPECT_EXIT(list_contains_item(int)(list, NULL), ::testing::ExitedWithCode(1),
              "NULL ptr error");
  EXPECT_EXIT(list_contains_item(int)(NULL, NULL), ::testing::ExitedWithCode(1),
              "NULL ptr error");
}