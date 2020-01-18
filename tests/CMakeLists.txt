set(test_target ${CMAKE_PROJECT_NAME}_test)

file(GLOB_RECURSE test_sources LIST_DIRECTORIES false *.cpp)

add_executable(${test_target} ${test_sources})
add_test(NAME ${test_target} COMMAND ${test_target})

get_filename_component(parent_dir ../ ABSOLUTE)
target_include_directories(${test_target} PRIVATE ${parent_dir}/cTorch/include)

target_link_libraries(${test_target} PUBLIC cTorch gtest)
