(This is a personal experimental project under development, not ready for prod.)

---
<p align="center">
  <img src="https://github.com/pcpLiu/cTorch/blob/master/logo.png" height="90">
</p>

<p align="center">
  <a href="https://github.com/pcpLiu/cTorch/actions">
    <img src="https://github.com/cTorch/cTorch/actions/workflows/build_linux.yml/badge.svg">
  </a>
  <a href="https://github.com/pcpLiu/cTorch/actions">
    <img src="https://github.com/cTorch/cTorch/actions/workflows/build_mac.yml/badge.svg">
  </a>
  <a href="https://codecov.io/gh/pcpLiu/cTorch">
    <img src="https://codecov.io/gh/pcpLiu/cTorch/branch/master/graph/badge.svg?token=G7rBTxAEAe" />
  </a>
  <a href="https://github.com/pcpLiu/cTorch/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue">
  </a>
</p>

# Introduction

cTorch is a light weight and flexible neural network inference library.
It is written in STD C11.

### Features

- **Prunable.** You can build cTorch with selective operators and backends
- **High-performance backends support.** cTorch supports several high performance backends: Intel MKL, CUDA etc

# Building & Installation

Building dependencies:

- CMake
- A C compiler: gcc or Clang

### Quick installation

```bash
$ git clone https://github.com/pcpLiu/cTorch
$ cd cTorch
$ mkdir build && cd build
$ cmake .. && make cTorch
$ sudo make install
```

### Backends

All operators in cTorch have a default implementation with standard C without dependency of any external lib.
cTorch also supports several high-performance backends: [OpenBLAS](), [Intel MKL](), [Apple]() and [CUDA]().

# License

Apache 2.0
