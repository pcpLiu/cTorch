#ifndef C_TORCH_LIBRARY_H
#define C_TORCH_LIBRARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cTorch/consts.h"
#include "cTorch/engine.h"
#include "cTorch/graph.h"
#include "cTorch/scheduler.h"
#include "cTorch/storage.h"

/**
 * Export internal files only in test mode
 */
#ifdef CTH_TEST_DEBUG
#include "cTorch/bit_array.h"
#include "cTorch/debug_util.h"
#include "cTorch/generic_array.h"
#include "cTorch/list_d.h"
#include "cTorch/node.h"
#include "cTorch/operator.h"
#include "cTorch/operators/op_list.h"
#include "cTorch/pool.h"
#include "cTorch/queue.h"
#include "cTorch/sharder.h"
#endif // CTH_TEST_DEBUG

#ifdef __cplusplus
}
#endif

#endif // C_TORCH_LIBRARY_H
