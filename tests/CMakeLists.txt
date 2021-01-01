project(cTorch_test LANGUAGES CXX)

# Profiling in linux
if(
    ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
)
    message("[cTorch_test] Append flag: -pg")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# libtorch
get_filename_component(PARENT_DIR ../ ABSOLUTE)
find_package(
    Torch REQUIRED
    HINTS ${PARENT_DIR}/third_party/libtorch
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Test files
file(GLOB_RECURSE test_sources LIST_DIRECTORIES false *_test.cpp)

# Utils
set(test_util_src
    # test_util.c
    test_util.cpp
    torch_util.cpp
)

set(test_target cTorch_test)
add_executable(${test_target}
    ${test_sources}
    ${test_util_src}
    main.cpp
)

add_test(NAME ${test_target} COMMAND ${test_target})
target_include_directories(${test_target} PRIVATE ${cTorch_project_SOURCE_DIR})
target_link_libraries(${test_target} PUBLIC cTorch ${TORCH_LIBRARIES} gtest)
# set_target_properties(${test_target} PROPERTIES CXX_STANDARD 14)

