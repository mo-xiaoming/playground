find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

add_library(scanner scanner.cpp)

llvm_map_components_to_libnames(llvm_libs support core)
find_package(spdlog REQUIRED)
target_link_libraries(scanner PUBLIC ${llvm_libs} spdlog::spdlog)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(scanner PUBLIC ${LLVM_DEFINITIONS_LIST})

add_library(test_main test.cpp)
target_link_libraries(test_main PUBLIC Catch2::Catch2)

add_executable(dl_test scanner_test.cpp)
target_link_libraries(dl_test PRIVATE scanner test_main)
catch_discover_tests(dl_test)
