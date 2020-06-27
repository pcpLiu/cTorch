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
