set(ops_mkl_all
    operators/mkl/op_list_mkl.c
    operators/mkl/op_abs_mkl.c
    operators/mkl/op_acos_mkl.c
    operators/mkl/op_add_mkl.c
    operators/mkl/op_square_mkl.c
    operators/mkl/op_mul_mkl.c
    operators/mkl/op_fmod_mkl.c
    operators/mkl/op_remainder_mkl.c
    operators/mkl/op_div_mkl.c
    operators/mkl/op_sqrt_mkl.c
    operators/mkl/op_rsqrt_mkl.c
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
