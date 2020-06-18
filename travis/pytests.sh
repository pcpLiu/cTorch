#!bin/bash

# build
cmake -DCMAKE_INSTALL_PREFIX:PATH=../tests/ops/pytests ..
cd cTorch
make && make install

# prepare cffi
cd ../../tests/ops/pytests
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib
export LIBRARY_PATH=$LIBRARY_PATH:lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:include
python3 cffi_util.py