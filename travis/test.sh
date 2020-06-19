#!bin/bash

# execute testing
cd tests

# tests can be run in paralllel
./cTorch_test --gtest_filter="-*MEMRECORD*"

# Involve with mem record count, run separatelly avoid of pthread abruption
./cTorch_test --gtest_filter="cTorchOperatorTest.testDeepFreeMEMRECORD"
./cTorch_test --gtest_filter="cTorchSharderTest.testTensorElewiseShardingMEMRECORD"
./cTorch_test --gtest_filter="cTorchSharderTest.testOperatorElewiseShardingMEMRECORD"


cd ..