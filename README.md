# cTorch

[![Build Status](https://travis-ci.com/pcpLiu/cTorch.svg?token=pqXTPvpuvJE34KJBhbJP&branch=master)](https://travis-ci.com/pcpLiu/cTorch)

# Introduction

cTorch is an inference engine for PyTorch models implemented with C.
It was developed for project [cTorch.js]().

# Backends

All operators are implemented with standard C.
However, cTorch also supports several backends by utilizing their performance APIs (Not all operators are supported).
When your machine supports one or more backends and you enable building against them, cTorch could execute ops on alternative backends.

## Backend dependencies

cTorch supports 6 backends: [x86 Intrinsics](), [ARM Intrinsics](), [OpenBLAS](), [Intel MKL](), [Accelerate]() and [CUDA]().
Based on your environment and needs, you should install one or all of them before heading to cTorch installation.

- **x86 Intrinsics**: (No installation needed) Support from SSE to AVX512
- **ARM Intrinsics**: (No installation needed)
- **OpenBLAS**: A high-performance BLAS implementation.
- **Intel MKL**:
- **Accelerate**: (No installation needed) Available on iOS && MacOS systems. It's a high-performance computational library offered by Apple.
- **CUDA**: cTorch supports CUDA 9

## Supported CPUs with Intrinsics Functions

- x86-64
  - Intel: Sandy Bridge & Later
  - AMD: Bulldozer & Zen
- ARM
  - 32 bits: v7-A/R, v8-A/R
  - 64 bits: v8-A

### Runtime backends V.S. built backend

When you are building cTorch, you could build against to multiple backends.
As you are using cTorch in your program, a specific operator will be executed on one backend.
If user specifies an unbuilt backend, runtime error will be raised.

### Automatically execution fallback

If you chose op to run on backend `X` while `X` does not support this operator, cTorch will
automatically switch execution to default implementation.

# Build instructions

Building dependencies:

- cMake
- gcc or Clang (g++ for running testing)
- POSIX Thread

### Build with selective operators

In `src/operators/CMakeLists.txt`, list operators into variable `ops_disabled`.
Those operators will be excluded from building.
All available operators' names are defined in `src/operators/op_list.h`.

# Contributors
