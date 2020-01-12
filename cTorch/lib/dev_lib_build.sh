#!/bin/bash

# jemalloc
cd ./jemalloc
./autogen.sh --with-jemalloc-prefix=je_  --disable-cxx
make
make INCLUDEDIR="../../include" install_include

# protobuf-c
cd ./protobuf-c