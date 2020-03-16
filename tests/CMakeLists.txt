set(test_target cTorch_test)

file(GLOB_RECURSE test_sources LIST_DIRECTORIES false *_test.cpp)

add_executable(${test_target} ${test_sources} main.cpp)
add_test(NAME ${test_target} COMMAND ${test_target})

target_include_directories(${test_target} PRIVATE ${cTorch_project_SOURCE_DIR})
target_link_libraries(${test_target} PUBLIC cTorch gtest)
