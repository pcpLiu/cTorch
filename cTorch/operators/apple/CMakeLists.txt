set(ops_apple_all
    operators/apple/op_list_apple.c
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
