#!bin/bash
cmake -DCMAKE_BUILD_TYPE=Debug  -DDEBUG_TEST=ON ..
make cTorch
make cTorch_test