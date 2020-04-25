project(cTorch_test LANGUAGES C CXX)

# Test files
file(GLOB_RECURSE test_sources LIST_DIRECTORIES false *_test.cpp)

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
add_test(NAME ${test_target} COMMAND ${test_target})
target_include_directories(${test_target} PRIVATE ${cTorch_project_SOURCE_DIR})
target_link_libraries(${test_target} PUBLIC cTorch gtest)
set_target_properties(${test_target} PROPERTIES CXX_STANDARD 11)
