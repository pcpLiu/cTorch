#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd DIR
mkdir installed

# Install OpenBLAS
cd OpenBLAS
make
make PREFIX=../installed install
