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
    operators/mkl/op_pow_mkl.c
    operators/mkl/op_exp_mkl.c
    operators/mkl/op_expm1_mkl.c
    operators/mkl/op_log_mkl.c
    operators/mkl/op_log2_mkl.c
    operators/mkl/op_log10_mkl.c
    operators/mkl/op_log1p_mkl.c
    operators/mkl/op_cos_mkl.c
    operators/mkl/op_sin_mkl.c
    operators/mkl/op_tan_mkl.c
    operators/mkl/op_asin_mkl.c
    operators/mkl/op_atan_mkl.c
    operators/mkl/op_atan2_mkl.c
    operators/mkl/op_cosh_mkl.c
    operators/mkl/op_sinh_mkl.c
    operators/mkl/op_tanh_mkl.c
    operators/mkl/op_erf_mkl.c
    operators/mkl/op_erfc_mkl.c
    operators/mkl/op_erfinv_mkl.c
    operators/mkl/op_floor_mkl.c
    operators/mkl/op_ceil_mkl.c
    operators/mkl/op_trunc_mkl.c
    operators/mkl/op_round_mkl.c
    operators/mkl/op_frac_mkl.c
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
