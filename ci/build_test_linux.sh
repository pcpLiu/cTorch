#!/bin/sh

#############################################################
#
# Build & test on Linux with backends:
#   - Default
#   - Intel MKL
#   - OpenBLAS
#
#############################################################

#############################################################
#
# Download libtorch
#
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip
mv libtorch third_party/

#############################################################
#
# Build
#

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

pwd
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDEBUG_TEST=ON \
    -DBACKEND_APPLE_ENABLE=OFF \
    -DBACKEND_MKL_ENABLE=OFF \
    -DCMAKE_PREFIX_PATH=../third_party/libtorch \
    ..

make ctorch_tests

#############################################################
#
# Test
#

# Any test fails, whole script execution fails.
# This is used to signal Github Actions
EXIT_STATUS=0

# tests can be run in paralllel and no Apple & CUDA backend
tests/ctorch_tests --gtest_filter="-*MEMRECORD" || EXIT_STATUS=$?

# Involve with mem record count, run separatelly avoid of pthread abruption
tests/ctorch_tests --gtest_filter="cTorchOperatorTest.testDeepFreeMEMRECORD" || EXIT_STATUS=$?
tests/ctorch_tests --gtest_filter="cTorchSharderTest.testTensorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
tests/ctorch_tests --gtest_filter="cTorchSharderTest.testOperatorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
tests/ctorch_tests --gtest_filter="cTorchListTest.testDeleteDataMEMRECORD" || EXIT_STATUS=$?

exit $EXIT_STATUS
