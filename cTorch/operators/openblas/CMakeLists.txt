set(ops_openblas_all
    PARENT_SCOPE
)

# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_openblas_all}
        EXCLUDE
        REGEX ^src/operators/openblas/${disabled_op}_openblas.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_openblas
    ${ops_openblas_all}
    PARENT_SCOPE
)