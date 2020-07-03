################################################################################
#
# List of disabled operators
#
#
set(ops_disabled)

################################################################################
#
# Add sub directory
#
add_subdirectory(default)    # ${ops_default}
add_subdirectory(x86)        # ${ops_x86}
add_subdirectory(cuda)       # ${ops_cuda}
add_subdirectory(mkl)        # ${ops_mkl}
add_subdirectory(openblas)   # ${ops_openblas}
add_subdirectory(apple)      # ${ops_apple}

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
