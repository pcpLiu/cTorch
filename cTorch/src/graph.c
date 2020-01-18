#include "graph.h"
#include <stdlib.h>

/*
  Get a input nodes list of a graph.
*/
ListCTorchNode *c_torch_graph_input_nodes(CTorchGraph *graph) {
  ListCTorchNode *head = (ListCTorchNode *)MALLOC(sizeof(ListCTorchNode));
  ListCTorchNode *curr_item = graph->nodes;
  while (curr_item != NULL && curr_item->data != NULL) {
    if (curr_item->data->inbound_nodes == NULL) {
      head = insert_into_list_ListCTorchNode(head, curr_item->data);
    }
    curr_item = curr_item->next_item;
  }
  return head;
}
