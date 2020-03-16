project(cTorch C)

################################################################################
#
# Build options

set(TARGET_CPU_ARCH "x86" CACHE STRING
    "Target cpu architecture. \
     Supported architectures: x86 or arm "
)

set(BACKEND_OPENBLAS_ENABLE ON CACHE BOOL
    "Buid backend with openBLAS"
)

set(BACKEND_MKL_ENABLE OFF CACHE BOOL
    "Buid backend with Intel MKL"
)

set(BACKEND_CUDA_ENABLE OFF CACHE BOOL
    "Buid backend with CUDA"
)

set(BACKEND_ACCELERATE_ENABLE OFF CACHE BOOL
    "Buid backend with Apple Accelerate"
)

################################################################################
#
# Preprocessors
#
if(${TARGET_CPU_ARCH} STREQUAL "x86")
    add_compile_definitions(BACKEND_CPU_X86)
elseif(${TARGET_CPU_ARCH} STREQUAL "arm")
    add_compile_definitions(BACKEND_CPU_ARM)
else()
    message(FATAL_ERROR "Unsupported CPU architecture: " ${TARGET_CPU_ARCH})
endif()

if(${BACKEND_OPENBLAS_ENABLE})
    add_compile_definitions(BACKEND_OPENBLAS)
endif()

if(${BACKEND_MKL_ENABLE})
    add_compile_definitions(BACKEND_MKL)
endif()

if(${BACKEND_CUDA_ENABLE})
    add_compile_definitions(BACKEND_CUDA)
endif()

if(${BACKEND_ACCELERATE_ENABLE})
    add_compile_definitions(BACKEND_ACCELERATE)
endif()

################################################################################
#
# Core sources
#
set(core_src
    engine.c
    graph.c
    node.c
    operator.c
    storage.c
    common.c
    plan.c
    consts.c
)

################################################################################
#
# Sub folders
#
add_subdirectory(operators) # ${operator_src}

################################################################################
#
# Setup cTorch target
#
add_library(cTorch
    STATIC
    ${core_src}
    ${operator_src}
)
target_include_directories(cTorch PUBLIC ${cTorch_project_SOURCE_DIR})
set_target_properties(cTorch PROPERTIES C_STANDARD 11)

################################################################################
#
# Install
#
include(GNUInstallDirs)
install(TARGETS cTorch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cTorch
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/cTorch
)
install(
    DIRECTORY ${cTorch_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/include
    FILES_MATCHING PATTERN "*.h"
)