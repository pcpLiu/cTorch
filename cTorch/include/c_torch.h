#ifndef C_TORCH_LIBRARY_H
#define C_TORCH_LIBRARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../src/consts.h"
#include "../src/graph.h"
#include "../src/list_d.h"
#include "../src/node.h"
#include "../src/operator.h"
#include "../src/plan.h"
#include "../src/storage.h"

// Append name space before identifier
#ifdef C_TORCH_NS
#define NS(name) C_TORCH_NS_##name
#else
#define NS(name) name
#endif

#ifdef __cplusplus
}
#endif

#endif // C_TORCH_LIBRARY_H
