<p align="center">
  <img src="https://github.com/pcpLiu/cTorch/blob/master/logo.png" height="90">
</p>

<p align="center">
  <a href="https://github.com/pcpLiu/cTorch/actions">
    <img src="https://github.com/pcpLiu/cTorch/workflows/build%20&%20test/badge.svg">
  </a>
  <a href="https://codecov.io/gh/pcpLiu/cTorch">
    <img src="https://codecov.io/gh/pcpLiu/cTorch/branch/master/graph/badge.svg?token=G7rBTxAEAe" />
  </a>
  <a>
    <img src="https://img.shields.io/badge/license-MIT-lightgrey">
  </a>
</p>

# I. Build instructions

Building dependencies:

- CMake
- gcc (g++ for running testing)
- POSIX Thread

### Quick build & install

```bash
$ git clone https://github.com/pcpLiu/cTorch
$ cd cTorch
$ mkdir build && cd build
$ cmake .. && make
$ sudo make install
```

### Build with selective operators

In `src/operators/CMakeLists.txt`, list operators into variable `ops_disabled`.
Those operators will be excluded from building.
All available operators' names are defined in `src/operators/op_list.h`.

# II. Backends

All operators in cTorch have a default implementation with standard C without dependency of any external lib.
cTorch also supports several high-performance backends: [x86 Intrinsics](), [ARM](), [OpenBLAS](), [Intel MKL](), [Apple]() and [CUDA]().
Based on your environment and needs, you should install one or all of them before heading to cTorch installation.

- [**x86 Intrinsics**](): support from SSE to AVX512
- [**ARM**](): [ARMNN]() and [ARM computer library]()
- [**OpenBLAS**](): A high-performance BLAS implementation
- [**Intel MKL**]():
- [**Apple**](): [Accelerate](https://developer.apple.com/documentation/accelerate), [Metal](https://developer.apple.com/documentation/metal) and [ML Compute](https://developer.apple.com/documentation/mlcompute) .
- [**CUDA**](): cTorch supports CUDA 9

### Runtime backends V.S. built backend

When you are building cTorch, you could build against to multiple backends.
As you are using cTorch in your program, an operator will be executed on one backend.
If user specifies an unbuilt backend, runtime error will be raised.

### Automatically execution fallback

If you chose op to run on backend `X` while `X` does not support this operator, cTorch will
automatically switch execution to default implementation.

# Contributors

This project exists thanks to all the people who contribute. [[Contribute](CONTRIBUTING.md)].
