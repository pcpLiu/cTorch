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
    -DBACKEND_MKL=ON \
    -DBACKEND_MKL_LIB_DIR=../third_party/intel_mkl/linux/lib/intel64 \
    -DBACKEND_MKL_INCLUDE_DIR=../third_party/intel_mkl/linux/include \
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

# tests can be run in paralllel
./cTorch_test --gtest_filter="-*MEMRECORD*" || EXIT_STATUS=$?

# Involve with mem record count, run separatelly avoid of pthread abruption
./cTorch_test --gtest_filter="cTorchOperatorTest.testDeepFreeMEMRECORD" || EXIT_STATUS=$?
./cTorch_test --gtest_filter="cTorchSharderTest.testTensorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./cTorch_test --gtest_filter="cTorchSharderTest.testOperatorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./cTorch_test --gtest_filter="cTorchListTest.testDeleteDataMEMRECORD" || EXIT_STATUS=$?

exit $EXIT_STATUS
