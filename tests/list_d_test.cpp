#include "cTorch/c_torch.h"
#include "tests/test_util.h"
#include "gtest/gtest.h"

// // for testFreeListDeep
// int *heap_int(int x) {
//   int *ptr = (int *)MALLOC(sizeof(int));
//   *ptr = x;
//   return ptr;
// }

// // for testFreeListDeep
// void free_deep_int(int *x) { FREE(x); }

cth_def_list_item(int);
def_list(int);
cth_impl_new_list_item_func(int);
cth_impl_new_list_func(int);
cth_impl_insert_list_func(int);
cth_impl_list_contains_data_func(int);
cth_impl_list_contains_item_func(int);
cth_impl_list_pop_func(int);
cth_impl_list_at_func(int);
cth_impl_free_list_func(int);
cth_impl_free_list_deep_func(int);
cth_impl_list_del_func(int);

TEST(cTorchListTest, testItemDefine) {
  int a = 3;
  CTHListItem(int) item = {
      .data = &a,
      .prev_item = NULL,
      .next_item = NULL,
  };
  EXPECT_EQ(*item.data, 3);
  EXPECT_EQ(item.prev_item, nullptr);
  EXPECT_EQ(item.next_item, nullptr);
}

TEST(cTorchListTest, testListDefine) {
  CTHList(int) list = {
      .size = 0,
      .head = NULL,
      .tail = NULL,
  };
  EXPECT_EQ(list.size, 0);
  EXPECT_EQ(list.head, nullptr);
  EXPECT_EQ(list.tail, nullptr);
}

TEST(cTorchListTest, testNewListItemFunc) {
  int a = 3;
  CTHListItem(int) *item = cth_new_list_item(int)(&a);
  EXPECT_EQ(*item->data, 3);
  EXPECT_EQ(item->prev_item, nullptr);
  EXPECT_EQ(item->next_item, nullptr);
}

TEST(cTorchListTest, testNewListItemFuncFail) {
  EXPECT_EXIT(cth_new_list_item(int)(NULL), ::testing::ExitedWithCode(1),
              "[.]*Pointer is NULL");
}

TEST(cTorchListTest, testNewListFunc) {
  CTHList(int) *list = cth_new_list(int)();
  EXPECT_EQ(list->size, 0);
  EXPECT_EQ(list->head, nullptr);
  EXPECT_EQ(list->tail, nullptr);
}

TEST(cTorchListTest, testNewListFuncStruct) {
  CTHList(CTHTensor) *list = cth_new_list(CTHTensor)();
  EXPECT_EQ(list->size, 0);
  EXPECT_EQ(list->head, nullptr);
  EXPECT_EQ(list->tail, nullptr);
}

TEST(cTorchListTest, testListInsertFunc) {
  int x[] = {1, 2, 3, 4};
  CTHList(int) *list = cth_new_list(int)();
  CTHListItem(int) *item_1 = cth_insert_list(int)(list, &x[0]);
  CTHListItem(int) *item_2 = cth_insert_list(int)(list, &x[1]);
  CTHListItem(int) *item_3 = cth_insert_list(int)(list, &x[2]);
  CTHListItem(int) *item_4 = cth_insert_list(int)(list, &x[3]);

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

TEST(cTorchListTest, testListInsertFuncFail) {
  int x[] = {1, 2, 3, 4};
  CTHList(int) *list = cth_new_list(int)();

  EXPECT_EXIT(cth_insert_list(int)(NULL, &x[0]), ::testing::ExitedWithCode(1),
              "[.]*Pointer is NULL");
  EXPECT_EXIT(cth_insert_list(int)(list, NULL), ::testing::ExitedWithCode(1),
              "[.]*Pointer is NULL");
  EXPECT_EXIT(cth_insert_list(int)(NULL, NULL), ::testing::ExitedWithCode(1),
              "[.]*Pointer is NULL");
}

TEST(cTorchListTest, testListContains) {
  int x[] = {1, 2, 3, 4, 5};
  CTHList(int) *list = cth_new_list(int)();
  CTHListItem(int) *item_1 = cth_insert_list(int)(list, &x[0]);
  CTHListItem(int) *item_2 = cth_insert_list(int)(list, &x[1]);
  CTHListItem(int) *item_3 = cth_insert_list(int)(list, &x[2]);
  CTHListItem(int) *item_4 = cth_insert_list(int)(list, &x[3]);

  CTHListItem(int) * found_item;
  found_item = cth_list_contains_data(int)(list, &x[0]);
  EXPECT_EQ(item_1, found_item);
  found_item = cth_list_contains_data(int)(list, &x[1]);
  EXPECT_EQ(item_2, found_item);
  found_item = cth_list_contains_data(int)(list, &x[2]);
  EXPECT_EQ(item_3, found_item);
  found_item = cth_list_contains_data(int)(list, &x[3]);
  EXPECT_EQ(item_4, found_item);
  found_item = cth_list_contains_data(int)(list, &x[4]);
  EXPECT_EQ(nullptr, found_item);

  cth_free_list(int)(list);
}

TEST(cTorchListTest, testListContainsFails) {
  int x[] = {1, 2, 3, 4};
  CTHList(int) *list = cth_new_list(int)();

  EXPECT_EXIT(cth_list_contains_data(int)(NULL, &x[0]),
              ::testing::ExitedWithCode(1), "[.]*Pointer is NULL");
  EXPECT_EXIT(cth_list_contains_data(int)(list, NULL),
              ::testing::ExitedWithCode(1), "[.]*Pointer is NULL");
  EXPECT_EXIT(cth_list_contains_data(int)(NULL, NULL),
              ::testing::ExitedWithCode(1), "[.]*Pointer is NULL");
}

TEST(cTorchListTest, testListContainsItem) {
  int x[] = {1, 2, 3, 4, 5};
  CTHList(int) *list = cth_new_list(int)();
  CTHListItem(int) *item_1 = cth_insert_list(int)(list, &x[0]);
  CTHListItem(int) *item_2 = cth_insert_list(int)(list, &x[1]);
  CTHListItem(int) *item_3 = cth_insert_list(int)(list, &x[2]);
  CTHListItem(int) *item_4 = cth_insert_list(int)(list, &x[3]);
  CTHListItem(int) *item_5 = cth_new_list_item(int)(&x[4]);

  bool found;
  found = cth_list_contains_item(int)(list, item_1);
  EXPECT_EQ(true, found);
  found = cth_list_contains_item(int)(list, item_2);
  EXPECT_EQ(true, found);
  found = cth_list_contains_item(int)(list, item_3);
  EXPECT_EQ(true, found);
  found = cth_list_contains_item(int)(list, item_4);
  EXPECT_EQ(true, found);
  found = cth_list_contains_item(int)(list, item_5);
  EXPECT_EQ(false, found);
}

TEST(cTorchListTest, testListContainsItemFails) {
  int x[] = {1, 2, 3, 4};
  CTHListItem(int) *item = cth_new_list_item(int)(&x[4]);
  CTHList(int) *list = cth_new_list(int)();

  EXPECT_EXIT(cth_list_contains_item(int)(NULL, item),
              ::testing::ExitedWithCode(1), "[.]*Pointer is NULL");
  EXPECT_EXIT(cth_list_contains_item(int)(list, NULL),
              ::testing::ExitedWithCode(1), "[.]*Pointer is NULL");
  EXPECT_EXIT(cth_list_contains_item(int)(NULL, NULL),
              ::testing::ExitedWithCode(1), "[.]*Pointer is NULL");
}

TEST(cTorchListTest, testListPop) {
  int x[] = {1, 2, 3, 4};
  CTHList(int) *list = cth_new_list(int)();
  CTHListItem(int) *item_1 = cth_insert_list(int)(list, &x[0]);
  CTHMemoryRecord *record_item_1 = cth_get_mem_record(item_1);
  CTHListItem(int) *item_2 = cth_insert_list(int)(list, &x[1]);
  CTHListItem(int) *item_3 = cth_insert_list(int)(list, &x[2]);
  CTHListItem(int) *item_4 = cth_insert_list(int)(list, &x[3]);
  int *pop = cth_list_pop(int)(list);
  EXPECT_EQ(*pop, x[0]);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_1->status);
  EXPECT_EQ(list->head, item_2);
  EXPECT_EQ(list->size, 3);
}

TEST(cTorchListTest, testListAt) {
  int x[] = {1, 2, 3, 4};
  CTHList(int) *list = cth_new_list(int)();
  CTHListItem(int) *item_1 = cth_insert_list(int)(list, &x[0]);
  CTHListItem(int) *item_2 = cth_insert_list(int)(list, &x[1]);
  CTHListItem(int) *item_3 = cth_insert_list(int)(list, &x[2]);
  CTHListItem(int) *item_4 = cth_insert_list(int)(list, &x[3]);

  EXPECT_EQ(*cth_list_at(int)(list, 0), x[0]);
  EXPECT_EQ(*cth_list_at(int)(list, 1), x[1]);
  EXPECT_EQ(*cth_list_at(int)(list, 2), x[2]);
  EXPECT_EQ(*cth_list_at(int)(list, 3), x[3]);

  EXPECT_EXIT(cth_list_at(int)(list, 4), ::testing::ExitedWithCode(1),
              "Error at func list_at:");
}

TEST(cTorchListTest, testFreeList) {

  int x[] = {1, 2, 3, 4};
  CTHList(int) *list = cth_new_list(int)();
  CTHMemoryRecord *record_list = cth_get_mem_record(list);

  CTHListItem(int) *item_1 = cth_insert_list(int)(list, &x[0]);
  CTHMemoryRecord *record_item_1 = cth_get_mem_record(item_1);

  CTHListItem(int) *item_2 = cth_insert_list(int)(list, &x[1]);
  CTHMemoryRecord *record_item_2 = cth_get_mem_record(item_2);

  CTHListItem(int) *item_3 = cth_insert_list(int)(list, &x[2]);
  CTHMemoryRecord *record_item_3 = cth_get_mem_record(item_3);

  CTHListItem(int) *item_4 = cth_insert_list(int)(list, &x[3]);
  CTHMemoryRecord *record_item_4 = cth_get_mem_record(item_4);

  cth_free_list(int)(list);

  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_list->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_1->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_2->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_3->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_4->status);
}

TEST(cTorchListTest, testFreeListDeep) {

  CTHList(int) *list = cth_new_list(int)();
  CTHMemoryRecord *record_list = cth_get_mem_record(list);

  int *data_1 = heap_int(1);
  CTHMemoryRecord *record_data_1 = cth_get_mem_record(data_1);
  CTHListItem(int) *item_1 = cth_insert_list(int)(list, data_1);
  CTHMemoryRecord *record_item_1 = cth_get_mem_record(item_1);

  int *data_2 = heap_int(2);
  CTHMemoryRecord *record_data_2 = cth_get_mem_record(data_2);
  CTHListItem(int) *item_2 = cth_insert_list(int)(list, data_2);
  CTHMemoryRecord *record_item_2 = cth_get_mem_record(item_2);

  int *data_3 = heap_int(3);
  CTHMemoryRecord *record_data_3 = cth_get_mem_record(data_3);
  CTHListItem(int) *item_3 = cth_insert_list(int)(list, data_3);
  CTHMemoryRecord *record_item_3 = cth_get_mem_record(item_3);

  int *data_4 = heap_int(4);
  CTHMemoryRecord *record_data_4 = cth_get_mem_record(data_4);
  CTHListItem(int) *item_4 = cth_insert_list(int)(list, data_4);
  CTHMemoryRecord *record_item_4 = cth_get_mem_record(item_4);

  cth_free_list_deep(int)(list);

  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_list->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_1->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_2->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_3->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_4->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_data_1->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_data_2->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_data_3->status);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_data_4->status);
}

TEST(cTorchListTest, testDeleteDataMEMRECORD) {
  // single thread execution
  CTHList(int) *list = cth_new_list(int)();
  int *data_1 = heap_int(1);
  CTHListItem(int) *item_1 = cth_insert_list(int)(list, data_1);
  CTHMemoryRecord *record_item_1 = cth_get_mem_record(item_1);

  int *data_2 = heap_int(2);
  cth_insert_list(int)(list, data_2);
  int *data_3 = heap_int(3);
  cth_insert_list(int)(list, data_3);

  int before_num = cth_get_num_unfree_records();
  cth_list_del(int)(list, data_1);
  int after_num = cth_get_num_unfree_records();
  EXPECT_EQ(after_num + 1, before_num);
  EXPECT_EQ(CTH_MEM_RECORD_STATUS_FREED, record_item_1->status);
}
