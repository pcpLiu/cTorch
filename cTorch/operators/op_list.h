#ifndef CTH_OP_LIST_H
#define CTH_OP_LIST_H

#ifdef BACKEND_CPU_X86
#include "cTorch/operators/x86/op_list.h"
#endif

#ifdef BACKEND_CPU_ARM
#include "cTorch/operators/arm/op_list.h"
#endif

#ifdef BACKEND_OPENBLAS
#include "cTorch/operators/openblas/op_list.h"
#endif

#ifdef BACKEND_MKL
#include "cTorch/operators/mkl/op_list.h"
#endif

#ifdef BACKEND_CUDA
#include "cTorch/operators/cuda/op_list.h"
#endif

#ifdef BACKEND_ACCELERATE
#include "cTorch/operators/accelerate/op_list.h"
#endif

#endif /* OP_LIST_H */
