################################################################################
#
# List of disabled operators
#
set(ops_disabled)

################################################################################
#
# Add sub directory
#
add_subdirectory(x86)        # ${ops_x86}
add_subdirectory(arm)        # ${ops_arm}
add_subdirectory(cuda)       # ${ops_cuda}
add_subdirectory(mkl)        # ${ops_mkl}
add_subdirectory(openblas)   # ${ops_openblas}
add_subdirectory(accelerate) # ${ops_accelerate}

################################################################################
#
# Export
#
set(operator_src
    ${ops_x86}
    ${ops_arm}
    ${ops_cuda}
    ${ops_accelerate}
    ${ops_openblas}
    ${ops_mkl}
    PARENT_SCOPE
)