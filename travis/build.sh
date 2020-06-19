#!bin/bash


####################################
#
# Build
#
cmake -DCMAKE_BUILD_TYPE=Debug  -DDEBUG_TEST=ON ..
make cTorch
make cTorch_test


####################################
#
# Test
#
cd tests

# tests can be run in paralllel
./cTorch_test --gtest_filter="-*MEMRECORD*"

# Involve with mem record count, run separatelly avoid of pthread abruption
./cTorch_test --gtest_filter="cTorchOperatorTest.testDeepFreeMEMRECORD"
./cTorch_test --gtest_filter="cTorchSharderTest.testTensorElewiseShardingMEMRECORD"
./cTorch_test --gtest_filter="cTorchSharderTest.testOperatorElewiseShardingMEMRECORD"
./cTorch_test --gtest_filter="cTorchListTest.testDeleteDataMEMRECORD"

cd ..