set(ops_mkl_all
    operators/mkl/op_list_mkl.c
    operators/mkl/op_abs_mkl.c
)

# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_mkl_all}
        EXCLUDE
        REGEX ^src/operators/mkl/${disabled_op}_mkl.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_mkl
    ${ops_mkl_all}
    PARENT_SCOPE
)
