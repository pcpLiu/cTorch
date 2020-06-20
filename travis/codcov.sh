#!bin/bash

# Skip gcov on osx or not using gcc
if [ "$TRAVIS_OS_NAME" = "osx" ] || [ "$TRAVIS_COMPILER" != "gcc" ]
then
    exit 0
fi


# generate gcov
cd cTorch/CMakeFiles/cTorch.dir
gcov *.gcno  -o .