#!/bin/sh

#############################################################
#
# Build & test on ARM cpu with backends:
#   - Default
#   - OpenBLAS
#   - Arm Neon
#
#############################################################


#############################################################
#
# Compile libtorch from source
#
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
python3 tools/build_libtorch.py

cd ..

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
    -DTARGET_CPU_ARCH=arm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDEBUG_TEST=ON \
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

# tests can be run in paralllel and no Apple & CUDA backend
./ctorch_tests --gtest_filter="-*MEMRECORD" || EXIT_STATUS=$?

# Involve with mem record count, run separatelly avoid of pthread abruption
./ctorch_tests --gtest_filter="cTorchOperatorTest.testDeepFreeMEMRECORD" || EXIT_STATUS=$?
./ctorch_tests --gtest_filter="cTorchSharderTest.testTensorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./ctorch_tests --gtest_filter="cTorchSharderTest.testOperatorElewiseShardingMEMRECORD" || EXIT_STATUS=$?
./ctorch_tests --gtest_filter="cTorchListTest.testDeleteDataMEMRECORD" || EXIT_STATUS=$?

exit $EXIT_STATUS
