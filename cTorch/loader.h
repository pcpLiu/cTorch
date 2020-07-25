#ifndef C_TORCH_LOADER_H
#define C_TORCH_LOADER_H

#include "cTorch/graph.h"

/*
  Load a graph from a fbs path.
  If any error happens, program will directly exit.

  Note: caller needs to release this plan manually.
*/
CTHGraph *load_graph(char *);

#endif /* LOADER_H */
