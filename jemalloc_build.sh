#!/bin/bash

cd ./lib/jemalloc

./autogen.sh --with-jemalloc-prefix=je_  --disable-cxx

make

make INCLUDEDIR="../../include" install_include