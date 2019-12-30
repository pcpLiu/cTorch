cmake_minimum_required(VERSION 3.0)
project(cTorch)
set(CMAKE_C_STANDARD 11)

##############################################
# options

option(USE_JEMALLOC  "Use jemalloc instead of system malloc" on)


set(link_libs)

##############################################
# Try use jemalloc

if(USE_JEMALLOC)
    message(STATUS "Building jemalloc locally...")
    execute_process(COMMAND
        bash "${PROJECT_SOURCE_DIR}/jemalloc_build.sh"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE jemalloc_build_status
        ERROR_VARIABLE jemalloc_build_err
        OUTPUT_QUIET
    )
    if("${jemalloc_build_status}" STREQUAL "0")
        message(STATUS "Successfully build jemalloc!")
        find_library(
            JEMALLOC_LIB_LOCAL_STATIC
            NAMES libjemalloc.a
            PATHS "${PROJECT_SOURCE_DIR}/lib/jemalloc/lib"
        )
        if(JEMALLOC_LIB_LOCAL_STATIC)
            message(STATUS "Found jemalloc! Will use it as default malloc.")
        else()
            message(WARNING "Cannot find local jemalloc. Will use available system malloc.")
            set(USE_JEMALLOC off)
        endif()
    else()
        message(WARNING
            "Failed to build jemalloc. Will use available system malloc."
        )
        set(USE_JEMALLOC off)
    endif()
endif()

##############################################
# Link backends

##############################################
# Build & Install cTorch

set(CTORCH_ROOT ${PROJECT_SOURCE_DIR}/src)

add_library(cTorch
    ${CTORCH_ROOT}/c_torch.h

    ${CTORCH_ROOT}/graph.c
    ${CTORCH_ROOT}/node.c
    ${CTORCH_ROOT}/storage.c
    ${CTORCH_ROOT}/engine.c
    ${CTORCH_ROOT}/operator.c
)
message(info ${JEMALLOC_LIB_LOCAL_STATIC})
target_link_libraries(cTorch ${JEMALLOC_LIB_LOCAL_STATIC})
target_include_directories(cTorch PRIVATE ${PROJECT_SOURCE_DIR}/include)