project(cTorch_test LANGUAGES C CXX)

# Profiling
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -pg)


# Test files
file(GLOB_RECURSE test_sources LIST_DIRECTORIES false *_test.cpp)

# Enable test debug option in cTorch building
add_compile_definitions(CTH_TEST_DEBUG)

# Utils
set(test_util_src
    test_util.c
)

set(test_target cTorch_test)
add_executable(${test_target}
    ${test_sources}
    ${test_util_src}
    main.cpp
)

# # gcov
# include(${PROJECT_SOURCE_DIR}/CMakeModules/CodeCoverage.cmake)
# APPEND_COVERAGE_COMPILER_FLAGS()
# setup_target_for_coverage_gcovr_xml(
#     NAME cTorch_test_coverage
#     EXECUTABLE ${test_target}
#     DEPENDENCIES ${test_target}
#     BASE_DIRECTORY "../cTorch"
# )

add_test(NAME ${test_target} COMMAND ${test_target})
target_include_directories(${test_target} PRIVATE ${cTorch_project_SOURCE_DIR})
target_link_libraries(${test_target} PUBLIC gtest cTorch)
set_target_properties(${test_target} PROPERTIES CXX_STANDARD 11)
