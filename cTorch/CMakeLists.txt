cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cTorch_lib DESCRIPTION "cTorch lib" LANGUAGES C)

set(cTorch_lib_VERSION 0.1.0)

set(CMAKE_C_STANDARD 11)

################################################################################
#
# pthread
#

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

################################################################################
#
# gprof
#

if(${DEBUG_TEST}
    AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
)
    message("[cTorch] Append flag: -pg --coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg --coverage")
endif()

################################################################################
#
# Source files
#

set(core_src
    bit_array.c
    config.c
    consts.c
    engine.c
    graph.c
    logger_util.c
    mem_util.c
    node.c
    operator.c
    params.c
    pool.c
    queue.c
    scheduler.c
    sharder.c
    storage.c
)

add_subdirectory(operators) # ${operator_src}

if(${DEBUG_TEST})
    set(core_src
        ${core_src}
        debug_util.c
    )
endif()

################################################################################
#
# Setup cTorch target
#

add_library(cTorch
    SHARED
    ${core_src}
    ${operator_src}
)
target_include_directories(cTorch PUBLIC ${cTorch_project_SOURCE_DIR})
set_target_properties(cTorch PROPERTIES C_STANDARD 11)
set_target_properties(cTorch PROPERTIES VERSION ${cTorch_lib_VERSION})

# link pthread
if(PTHREAD_ENABLE)
    target_link_libraries(cTorch PUBLIC Threads::Threads)
endif()

# link gcov for test on linux
if(
    ${DEBUG_TEST}
    AND  ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
)
    message("[cTorch]: Will link gcov")
    target_link_libraries(cTorch PUBLIC gcov)
endif()


################################################################################
#
# Backends link
#

if(${TARGET_CPU_ARCH} STREQUAL "x86")
elseif(${TARGET_CPU_ARCH} STREQUAL "arm")
else()
    message(FATAL_ERROR "Unsupported CPU architecture: " ${TARGET_CPU_ARCH})
endif()

# Openblas
if(${BACKEND_OPENBLAS_ENABLE})
endif()

# MKL
if(${BACKEND_MKL_ENABLE})
    find_library(
        MKL_CORE_LIB
        NAMES mkl_core
        PATHS ${BACKEND_MKL_LIB_DIR}
        REQUIRED
    )
    find_library(
        MKL_INTERFACE_LIB
        NAMES mkl_intel_ilp64
        PATHS ${BACKEND_MKL_LIB_DIR}
        REQUIRED
    )
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        find_library(
            MKL_THREAD_LIB
            NAMES mkl_gnu_thread
            PATHS ${BACKEND_MKL_LIB_DIR}
            REQUIRED
        )
    else()
        find_library(
            MKL_THREAD_LIB
            NAMES mkl_tbb_thread
            PATHS ${BACKEND_MKL_LIB_DIR}
            REQUIRED
        )
    endif()
    find_package(OpenMP REQUIRED)
    set(${CMAKE_C_FLAGS} "${CMAKE_C_FLAGS} -m64")
    target_include_directories(cTorch PRIVATE ${BACKEND_MKL_INCLUDE_DIR})
    message(
        "[cTorch]: Append link libraries for MKL backend: "
        "${MKL_INTERFACE_LIB}, ${MKL_CORE_LIB}, ${MKL_THREAD_LIB}, OpenMP::OpenMP_C"
    )
    # !Link order matters!
    target_link_libraries(
        cTorch PUBLIC
        ${MKL_INTERFACE_LIB}
        ${MKL_CORE_LIB}
        ${MKL_THREAD_LIB}
        OpenMP::OpenMP_C
    )
endif()

# CUDA
if(${BACKEND_CUDA_ENABLE})
endif()

# Apple
if(${BACKEND_APPLE_ENABLE})
    find_library(
        APPLE_ACCELERATE_FRAMEWORK
        NAMES Accelerate accelerate
        PATHS "/Library"
        REQUIRED
    )
    message(
        "[cTorch]: Append link libraries for Apple backend: "
        "${APPLE_ACCELERATE_FRAMEWORK}"
    )
    target_link_libraries(
        cTorch PUBLIC
        ${APPLE_ACCELERATE_FRAMEWORK}
    )
endif()


################################################################################
#
# Install cTorch
#

include(GNUInstallDirs)
install(TARGETS cTorch
    EXPORT cTorchTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cTorch
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/cTorch
)
install(
    DIRECTORY ${cTorch_lib_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
