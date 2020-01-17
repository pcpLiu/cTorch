################################################################################
# Lib

add_library(cTorch SHARED
    src/engine.c
    src/graph.c
    src/list.c
    src/node.c
    src/operator.c
    src/storage.c
)
target_include_directories(cTorch PRIVATE src)
set_target_properties(cTorch PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(cTorch PROPERTIES PUBLIC_HEADER include/c_torch.h)

################################################################################
# Lib-dev

file(GLOB_RECURSE sources LIST_DIRECTORIES true *.h *.c)
add_library(cTorch-dev STATIC ${sources})
set_target_properties(cTorch-dev PROPERTIES VERSION ${PROJECT_VERSION})

