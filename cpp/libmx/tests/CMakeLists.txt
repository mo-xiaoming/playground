find_package(GTest)
include(GoogleTest)

add_executable(tests
	mx_test.cpp
)
target_compile_definitions(tests PRIVATE IS_TEST_BUILD)
set_target_properties(tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(tests PRIVATE ${COMMON_FLAGS})
target_link_libraries(tests PRIVATE GTest::Main mx)
gtest_discover_tests(tests)
