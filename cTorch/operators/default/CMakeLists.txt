################################################################################
#
# All ops
#
set(ops_default_all
    operators/default/op_list.c
    operators/default/op_abs_default.c
    operators/default/op_acos_default.c
    operators/default/op_add_default.c
    operators/default/op_addcdiv_default.c
    operators/default/op_addcmul_default.c
    operators/default/op_angle_default.c
    operators/default/op_asin_default.c
    operators/default/op_atan_default.c
    operators/default/op_atan2_default.c
    operators/default/op_bitwise_not_default.c
    operators/default/op_bitwise_and_default.c
    operators/default/op_bitwise_or_default.c
    operators/default/op_bitwise_xor_default.c
    operators/default/op_ceil_default.c
    operators/default/op_clamp_default.c
    operators/default/op_cos_default.c
    operators/default/op_cosh_default.c
    operators/default/op_div_default.c
    operators/default/op_digamma_default.c
    operators/default/op_erf_default.c
    operators/default/op_erfc_default.c
    operators/default/op_exp_default.c
    operators/default/op_expm1_default.c
    operators/default/op_floor_default.c
    operators/default/op_floor_divide_default.c
    operators/default/op_fmod_default.c
    operators/default/op_frac_default.c
    operators/default/op_lerp_default.c
)

################################################################################
#
# Remove disabled operators
#
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_default_all}
        EXCLUDE
        REGEX ^operators/default/${disabled_op}_default.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_default
    ${ops_default_all}
    PARENT_SCOPE
)
