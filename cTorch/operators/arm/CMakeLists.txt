set(ops_arm_all
    PARENT_SCOPE
)

# support float16 on ARM
if(BACKEND_CPU_ARM)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfp16-format=ieee")
endif()

# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_arm_all}
        EXCLUDE
        REGEX ^src/operators/arm/${disabled_op}_arm.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_arm
    ${ops_arm_all}
    PARENT_SCOPE
)