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
It is written in pure C11 and compatible with all operators of PyTorch.

### Features

- **Prunable.** You can build cTorch with selective operators and backends
- **High-performance backends support.** cTorch supports several high performance backends: Intel MKL, CUDA etc
- **One click convert.** We offer a python tool [Cerberus
  ](https://github.com/pcpLiu/Cerberus) converting the existing PyTorch model to the format that can be exuecuted by cTorch

Check more at [FAQ]() page.

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

See [Building and Installation Guide]() for more information.

# Parallel model

cTorch is designed for a typical single-process multi-threading deployment environment.
In terms of parallelism, there are two levels:

1. Inter-op:
2. Intra-op:

# License

cTorch is [MIT](https://github.com/pcpLiu/cTorch/blob/master/LICENSE) licensed.

# Releases and Contributing

cTorch has a bi-weekly release focusing on bug-fix.
We also have a 90-day major release walking with PyTorch.

We appreciate all contributions.
Take a loot at our [Contribution Guide]() see how you can help make cTorch better!
