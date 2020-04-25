cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cTorch_project)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

enable_testing()
add_subdirectory(cTorch)
add_subdirectory(lib/googletest)
add_subdirectory(tests)
