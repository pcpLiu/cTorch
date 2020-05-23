#include "cTorch/scheduler.h"
#include "cTorch/mem_util.h"

CTorchScheduler *cth_new_scheduler(CTorchConfig *config) {
  CTorchScheduler *scheduler = MALLOC(sizeof(CTorchScheduler));
  scheduler->done_queue = cth_new_queue();
  scheduler->ready_queue = cth_new_queue();

  return scheduler;
}