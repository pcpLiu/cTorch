#include "graph.h"
#include <stdlib.h>

/*
  Get a input nodes list of a graph.
*/
CTorchNodeList *c_torch_graph_input_nodes(CTorchGraph *graph) {
  CTorchNodeList *head = MALLOC(sizeof(CTorchNodeList));
  CTorchNodeList *curr_item = graph->nodes;
  while (curr_item != NULL && curr_item->data != NULL) {
    if (curr_item->data->in_bound_nodes == NULL) {
      head = insert_into_list_CTorchNodeList(head, curr_item->data);
    }
    curr_item = curr_item->next_item;
  }
  return head;
}
