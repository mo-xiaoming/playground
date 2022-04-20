find_package(Catch2 CONFIG REQUIRED)
include(Catch)

add_executable(test_buffers test_main.cpp test_buffers.cpp)
target_link_libraries(test_buffers PRIVATE Catch2::Catch2 buffers)
catch_discover_tests(test_buffers)
add_lint(test_buffers)
