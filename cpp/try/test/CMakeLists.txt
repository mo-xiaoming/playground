find_package(fmt REQUIRED)

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

find_package(Catch2 REQUIRED)
include(Catch)

function(add_my_test name sources)
    add_executable(${name} test_main.cpp ${sources})
    target_link_libraries(${name} PRIVATE Catch2::Catch2 fmt::fmt)
    catch_discover_tests(${name})
endfunction()

add_my_test(test_catch2 test_catch2.cpp)
add_my_test(test_fmt test_fmt.cpp)
