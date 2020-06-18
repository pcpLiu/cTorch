#!bin/bash

# execute testing
cmake -DCMAKE_BUILD_TYPE=Debug ..
cd tests
./cTorch_test

# generate gcov
cd ../cTorch/CMakeFiles/cTorch.dir
gcov *.gcno  -o .