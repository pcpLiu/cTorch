set(ops_apple_all
    operators/apple/op_list_apple.c
    operators/apple/op_ceil_apple.c
    operators/apple/op_floor_apple.c
    operators/apple/op_div_apple.c
    operators/apple/op_abs_apple.c
    operators/apple/op_fmod_apple.c
    operators/apple/op_remainder_apple.c
    operators/apple/op_trunc_apple.c
    operators/apple/op_round_apple.c
    operators/apple/op_rsqrt_apple.c
    operators/apple/op_sqrt_apple.c
    operators/apple/op_reciprocal_apple.c
    operators/apple/op_exp_apple.c
    operators/apple/op_expm1_apple.c
    operators/apple/op_log_apple.c
    operators/apple/op_log1p_apple.c
    operators/apple/op_log2_apple.c
    operators/apple/op_log10_apple.c
    operators/apple/op_pow_apple.c
    operators/apple/op_sin_apple.c
    operators/apple/op_cos_apple.c
    operators/apple/op_tan_apple.c
    operators/apple/op_asin_apple.c
    operators/apple/op_acos_apple.c
    operators/apple/op_actan_apple.c
    operators/apple/op_actan2_apple.c
    operators/apple/op_sinh_apple.c
    operators/apple/op_cosh_apple.c
    operators/apple/op_tanh_apple.c
)

# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_apple_all}
        EXCLUDE
        REGEX ^src/operators/apple/${disabled_op}_apple.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_apple
    ${ops_apple_all}
    PARENT_SCOPE
)
