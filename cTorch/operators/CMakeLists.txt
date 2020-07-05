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
add_subdirectory(x86)        # ${ops_x86}

if(${BACKEND_CUDA_ENABLE})
    message("[cTorh]: Will build CUDA operators")
    add_subdirectory(cuda)       # ${ops_cuda}
endif()

if(${BACKEND_MKL_ENABLE})
    message("[cTorh]: Will build Intel MKL operators")
    add_subdirectory(mkl)        # ${ops_mkl}
endif()

if(${BACKEND_OPENBLAS_ENABLE})
    message("[cTorh]: Will build OpenBLAS operators")
    add_subdirectory(openblas)   # ${ops_openblas}
endif()

if(${BACKEND_APPLE_ENABLE})
    message("[cTorh]: Will build Apple operators")
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
