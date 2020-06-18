#!bin/bash
cmake -DCMAKE_BUILD_TYPE=Debug  ..
make cTorch
make cTorch_test