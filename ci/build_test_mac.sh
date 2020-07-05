#!/bin/sh

#############################################################
#
# Build & test on Linux with backends:
#   - Default
#   - Intel MKL
#   - Apple
#   - OpenBLAS
#
#############################################################


#############################################################
#
# Build
#

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# install Intel tbb & libomp for MKL backend
brew install tbb
brew install libomp

cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDEBUG_TEST=ON \
    -DBACKEND_APPLE_ENABLE=ON \
    -DBACKEND_MKL_ENABLE=ON \
    -DBACKEND_MKL_LIB_DIR=../third_party/intel_mkl/mac/lib \
    -DBACKEND_MKL_INCLUDE_DIR=../third_party/intel_mkl/mac/include \
    ..

make cTorch_test

#############################################################
#
# Test
#

cd tests

# Any test fails, whole script execution fails.
# This is used to signal Github Actions
EXIT_STATUS=0

# tests can be run in paralllel and no CUDA backend
./cTorch_test --gtest_filter="-*MEMRECORD:*CUDA" || EXIT_STATUS=$?

# Involve with mem record count, run separatelly avoid of pthread abruption
./cTorch_test --gtest_filter="cTorchOperatorTest.testDeepFreeMEMRECORD" || EXIT_STATUS=$?
./cTorch_test --gtest_filter="cTorchSharderTest.testTensorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./cTorch_test --gtest_filter="cTorchSharderTest.testOperatorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./cTorch_test --gtest_filter="cTorchListTest.testDeleteDataMEMRECORD" || EXIT_STATUS=$?

exit $EXIT_STATUS
