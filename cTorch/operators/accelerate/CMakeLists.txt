set(ops_accelerate_all
    PARENT_SCOPE
)

# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_accelerate_all}
        EXCLUDE
        REGEX ^src/operators/accelerate/${disabled_op}_accelerate.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_accelerate
    ${ops_accelerate_all}
    PARENT_SCOPE
)