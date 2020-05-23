project(cTorch_lib VERSION 0.1.0 DESCRIPTION "cTorch lib" LANGUAGES C)

################################################################################
#
# Build options
#
set(TARGET_CPU_ARCH "x86" CACHE STRING
    "Target cpu architecture. \
     Supported architectures: x86 or arm "
)

set(BACKEND_OPENBLAS_ENABLE ON CACHE BOOL
    "Build backend with openBLAS"
)

set(BACKEND_MKL_ENABLE OFF CACHE BOOL
    "Build backend with Intel MKL"
)

set(BACKEND_CUDA_ENABLE OFF CACHE BOOL
    "Build backend with CUDA"
)

set(BACKEND_ACCELERATE_ENABLE OFF CACHE BOOL
    "Build backend with Apple Accelerate"
)

set(PTHREAD_ENABLE ON CACHE BOOL
    "Build with pthread support. Default ON."
)

set(DEBUG_TEST ON CACHE BOOL
    "Enable debug test options. Default ON. Option should only be enabled for running tests."
)

################################################################################
#
# Backends preprocessors
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

if(${DEBUG_TEST})
    add_compile_definitions(CTH_TEST_DEBUG)
    message(
        WARNING
        "DEBUG_TEST is turned ON. Option should only be enabled for running tests."
    )
endif()

################################################################################
#
# Core sources
#
set(core_src
    config.c
    consts.c
    debug_util.c
    engine.c
    graph.c
    logger_util.c
    mem_util.c
    node.c
    operator.c
    pool.c
    queue.c
    scheduler.c
    sharder.c
    storage.c
)

################################################################################
#
# Import pthread
#
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

add_compile_definitions(USE_PTHREAD)
set(core_src
    ${core_src}
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

# link pthread
if(PTHREAD_ENABLE)
    target_link_libraries(cTorch Threads::Threads)
endif()

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