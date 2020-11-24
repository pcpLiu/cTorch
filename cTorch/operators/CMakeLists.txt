################################################################################
#
# List of disabled operators
#
set(ops_disabled)

################################################################################
#
# Add backend operators based on switches
#
add_subdirectory(default)    # ${ops_default}

if (${TARGET_CPU_ARCH} STREQUAL "x86")
    message("[cTorch]: Will optimize for x86")
    add_subdirectory(x86)        # ${ops_x86}
endif()

if (${TARGET_CPU_ARCH} STREQUAL "arm")
    message("[cTorch]: Will optimize for ARM")
    add_subdirectory(arm)        # ${ops_arm}
endif()

if(${BACKEND_CUDA_ENABLE})
    message("[cTorch]: Will build against CUDA backend")
    add_subdirectory(cuda)       # ${ops_cuda}
endif()

if(${BACKEND_MKL_ENABLE})
    message("[cTorch]: Will build against Intel MKL backend")
    add_subdirectory(mkl)        # ${ops_mkl}
endif()

if(${BACKEND_OPENBLAS_ENABLE})
    message("[cTorch]: Will build against OpenBLAS backend")
    add_subdirectory(openblas)   # ${ops_openblas}
endif()

if(${BACKEND_APPLE_ENABLE})
    message("[cTorch]: Will build against Apple backend")
    add_subdirectory(apple)      # ${ops_apple}
endif()

################################################################################
#
# Export
#
set(operator_src
    ${ops_default}
    ${ops_x86}
    ${ops_arm}
    ${ops_cuda}
    ${ops_apple}
    ${ops_openblas}
    ${ops_mkl}
    PARENT_SCOPE
)
