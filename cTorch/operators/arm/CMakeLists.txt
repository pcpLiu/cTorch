set(ops_arm_all)



# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_arm_all}
        EXCLUDE
        REGEX ^src/operators/mkl/${disabled_op}_arm.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_arm
    ${ops_mkl_all}
    PARENT_SCOPE
)
