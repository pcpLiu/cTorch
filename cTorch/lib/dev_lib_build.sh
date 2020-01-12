#!/bin/bash

# jemalloc
cd ./jemalloc
./autogen.sh --with-jemalloc-prefix=je_  --disable-cxx
make && make install

# protobuf
cd ../protobuf
./autogen.sh && ./configure && make && make install

# protobuf-c
cd ../protobuf-c
./autogen.sh && ./configure && make && make install