#!bin/bash

# generate gcov
cd cTorch/CMakeFiles/cTorch.dir
gcov *.gcno  -o .

# upload
bash <(curl -s https://codecov.io/bash) -t a304e6b7-f321-4bfc-b622-27bf15ed93b4
