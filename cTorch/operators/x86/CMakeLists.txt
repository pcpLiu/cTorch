################################################################################
#
# All ops
#
set(ops_x86_all
    operators/x86/op_list.c
)

################################################################################
#
# Remove disabled operators
#
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_x86_all}
        EXCLUDE
        REGEX ^operators/default/${disabled_op}_x86.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_x86
    ${ops_x86_all}
    PARENT_SCOPE
)