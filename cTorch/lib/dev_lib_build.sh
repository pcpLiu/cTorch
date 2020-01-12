#!/bin/bash

# jemalloc
cd ./jemalloc
./autogen.sh --with-jemalloc-prefix=je_  --disable-cxx
make && make install

# flatcc
cd ../flatcc
scripts/build.sh
