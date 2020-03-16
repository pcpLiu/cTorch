# cTorch

[![Build Status](https://travis-ci.com/pcpLiu/cTorch.svg?token=pqXTPvpuvJE34KJBhbJP&branch=master)](https://travis-ci.com/pcpLiu/cTorch)

# Introduction

cTorch is an inference engine for PyTorch models implemented with C.
It was developed for project [cTorch.js]().

# Backends

## Supported CPUs

- x86-64
  - Intel: Sandy Bridge & Later
  - AMD: Bulldozer & Zen
- ARM
  - 32 bits: v7-A/R, v8-A/R
  - 64 bits: v8-A

## Backend dependencies

cTorch supports 4 backends: [OpenBLAS](), [Intel MKL](), [Accelerate]() and [CUDA]().
Based on your environment and needs, you should install one or all of them before heading to cTorch installation.

- **OpenBLAS (default backend)**: We added OpenBLAS as a submodule, if you chose OpenBLAS as backend, you don't need to do extra installation
- **Intel MKL**: iN
- **Accelerate**: available on iOS && MacOS systems and you don't need to install them. It's a high-performance computational library offered by Apple
- **CUDA**: we support CUDA 8 & 9

### Runtime backends V.S. built backend

When you are building cTorch, you could build against to multiple backends.
As you are using cTorch in your program, a specific operator will be executed on one backend.
If user specifies an unavailable backend, runtime error will be raised.

# Build instructions

Building tools:

- cMake
- A C compiler (GCC or Clang)

### Build with selective operators

In `src/operators/CMakeLists.txt`, list operators into variable `ops_disabled`.
Those operators will be excluded from building.
All available operators' names are defined in `src/operators/op_list.h`.
