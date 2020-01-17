cmake_minimum_required(VERSION 3.0)
project(cTorch VERSION 0.1.0 DESCRIPTION "cTorch lib" LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# add_library(c_torch SHARED
#     cTorch/src/engine.c
#     cTorch/src/graph.c
#     cTorch/src/list.c
#     cTorch/src/node.c
#     cTorch/src/operator.c
#     cTorch/src/storage.c
# )

# target_include_directories(c_torch PRIVATE cTorch/src)

# set_target_properties(c_torch PROPERTIES VERSION ${PROJECT_VERSION})
# set_target_properties(c_torch PROPERTIES PUBLIC_HEADER include/c_torch.h)

add_subdirectory(cTorch)
add_subdirectory(tests)
add_subdirectory(lib/googletest)

################################################################################
# Install

# include(GNUInstallDirs)
# install(TARGETS c_torch
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )