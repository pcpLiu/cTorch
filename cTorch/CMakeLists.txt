################################################################################
# Lib

add_library(cTorch SHARED
    # src/engine.c
    src/graph.c
    src/node.c
    src/operator.c
    src/storage.c
    src/common.c
    src/plan.c
)
target_include_directories(cTorch PRIVATE src)
set_target_properties(cTorch PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(cTorch PROPERTIES PUBLIC_HEADER include/c_torch.h)
