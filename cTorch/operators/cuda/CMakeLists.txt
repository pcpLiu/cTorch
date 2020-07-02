set(ops_cuda_all
    operators/cuda/op_list_cuda.c
)

# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_cuda_all}
        EXCLUDE
        REGEX ^src/operators/cuda/${disabled_op}_cuda.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_cuda
    ${ops_cuda_all}
    PARENT_SCOPE
)
