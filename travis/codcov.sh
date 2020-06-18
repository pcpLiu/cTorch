#!bin/bash

# Skip gcov on osx
if [ "$TRAVIS_OS_NAME" = "osx" ]; then exit 0 ; fi


# generate gcov
cd ../cTorch/CMakeFiles/cTorch.dir
gcov *.gcno  -o .