#!/bin/sh

cd build/tests

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
