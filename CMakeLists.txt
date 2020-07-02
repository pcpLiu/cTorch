cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(cTorch_project)

enable_testing()
add_subdirectory(cTorch)
add_subdirectory(third_party/googletest)
add_subdirectory(tests)
