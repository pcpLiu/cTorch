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
# Download libtorch
#
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip
unzip libtorch-macos-1.5.1.zip
mv libtorch third_party/

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

CC=clang CXX=clang++ \
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDEBUG_TEST=ON \
    -DBACKEND_APPLE_ENABLE=ON \
    -DBACKEND_MKL_ENABLE=ON \
    -DBACKEND_MKL_LIB_DIR=../third_party/intel_mkl/mac/lib \
    -DBACKEND_MKL_INCLUDE_DIR=../third_party/intel_mkl/mac/include \
    -DCMAKE_PREFIX_PATH=../third_party/libtorch \
    ..

make ctorch_tests

#############################################################
#
# Test
#

cd tests

# Any test fails, whole script execution fails.
# This is used to signal Github Actions
EXIT_STATUS=0

# tests can be run in paralllel and no CUDA backend
./ctorch_tests --gtest_filter="-*MEMRECORD" || EXIT_STATUS=$?

# Involve with mem record count, run separatelly avoid of pthread abruption
./ctorch_tests --gtest_filter="cTorchOperatorTest.testDeepFreeMEMRECORD" || EXIT_STATUS=$?
./ctorch_tests --gtest_filter="cTorchSharderTest.testTensorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./ctorch_tests --gtest_filter="cTorchSharderTest.testOperatorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./ctorch_tests --gtest_filter="cTorchListTest.testDeleteDataMEMRECORD" || EXIT_STATUS=$?

exit $EXIT_STATUS
