cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cTorch_lib DESCRIPTION "libctorch" LANGUAGES C)

set(cTorch_lib_VERSION 0.1.0)

set(CMAKE_C_STANDARD 11)

################################################################################
#
# pthread
#
if(
    ${ENABLE_SHARDING}
    OR ${DEBUG_TEST}
)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    find_package(Threads REQUIRED)
endif()

################################################################################
#
# gprof in debug mode
#
if(${DEBUG_TEST}
    AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
    AND (
        CMAKE_C_COMPILER_ID STREQUAL "GNU"
        OR
        CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    )
)
    message("[cTorch] Append flag: -pg --coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg --coverage")
endif()

################################################################################
#
# ARM half-precision flags
# https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html
#
if(${TARGET_CPU_ARCH} STREQUAL "arm")
    message("[cTorch] Append flag: -mfp16-format=alternative -mfpu=neon-fp16")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfp16-format=alternative -mfpu=neon-fp16 -mfloat-abi=hard")
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

add_library(ctorch
    STATIC
    ${core_src}
    ${operator_src}
)
target_include_directories(ctorch PUBLIC ${cTorch_project_SOURCE_DIR})
set_target_properties(ctorch PROPERTIES C_STANDARD 11)
set_target_properties(ctorch PROPERTIES VERSION ${cTorch_lib_VERSION})

# link pthread
if(
    ${ENABLE_SHARDING}
    OR ${DEBUG_TEST}
)
    target_link_libraries(ctorch PUBLIC Threads::Threads)
endif()

# link gcov for test on linux
if(
    ${DEBUG_TEST}
    AND  ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
    AND (
        CMAKE_C_COMPILER_ID STREQUAL "GNU"
        OR
        CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    )
)
    message("[cTorch]: Will link gcov")
    target_link_libraries(ctorch PUBLIC gcov)
endif()


################################################################################
#
# Backends link
#

# CPU
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

    # This part is highly customizable. Here we config for few common situations.
    # Check Inte's Link advisor for your hardware environment.

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(MKL_ARCH "intel64")
        set(${CMAKE_C_FLAGS} "${CMAKE_C_FLAGS} -m64")
        set(MKL_LIB_DIR "${BACKEND_MKL_ROOT_DIR}/lib/intel64")
    else()
        set(MKL_ARCH "ia32")
        set(${CMAKE_C_FLAGS} "${CMAKE_C_FLAGS} -m32")
        set(MKL_LIB_DIR "${BACKEND_MKL_ROOT_DIR}/lib/ia32")
    endif()

    if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "GNU")
        set(MKL_THREADING "gnu_thread")
        set(MKL_INTERFACE "lp64")
    else(CMAKE_C_COMPILER_ID MATCHES "Intel" OR CMAKE_C_COMPILER_ID MATCHES "MSVC")
        set(MKL_THREADING "intel_thread")
        set(MKL_INTERFACE "ilp64")
    endif()

    set(MKL_LINK "dynamic")

    find_package(
        MKL
        PATHS "${BACKEND_MKL_ROOT_DIR}/lib"
        REQUIRED
    )

    # Both ctorch & libtorch (used in tests) will find MKL pakcage.
    # Use this variable to avoid double configing
    if(${DEBUG_TEST})
        set(MKL_FOUND PARENT_SCOPE)
    endif()

    target_include_directories(
        ctorch PUBLIC
        ${MKL_INCLUDE}
    )

    target_link_libraries(
        ctorch PUBLIC
       $<LINK_ONLY:MKL::MKL>
    )

    set_target_properties(ctorch PROPERTIES ENVIRONMENT "${MKL_ENV}")

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
        ctorch PUBLIC
        ${APPLE_ACCELERATE_FRAMEWORK}
    )
endif()


################################################################################
#
# Install cTorch
#

include(GNUInstallDirs)
install(TARGETS ctorch
    EXPORT cTorchTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ctorch
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/ctorch
)
install(
    DIRECTORY ${cTorch_lib_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
