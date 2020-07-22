#ifndef CTH_CONFIG_H
#define CTH_CONFIG_H

#include "cTorch/consts.h"

/**
 * Execution config
 */
typedef struct CTHConfig {
  cth_thread_n_t num_workers; /* No. of workers in the pool */
} CTHConfig;

#endif /* CONFIG_H */
