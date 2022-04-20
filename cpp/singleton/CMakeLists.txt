cmake_minimum_required(VERSION 3.17)
if(NOT VCPKG_HOME)
	message(FATAL_ERROR "VCPKG_HOME is undefined")
endif()
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_HOME}/scripts/buildsystems/vcpkg.cmake")
project(TestLogger VERSION 1.0.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fno-rtti -Wfatal-errors")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(cxx_strict_flags "-Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
elseif(CMAKE_COMPILER_IS_GNUCXX)
	set(cxx_strict_flags "-Wcast-align=strict -Wcast-qual -Wconversion -Wdouble-promotion -Wduplicated-branches -Wduplicated-cond -Wfloat-equal -Wformat=2 -Wlogical-op -Wmisleading-indentation -Wmissing-declarations -Wmissing-include-dirs -Wnon-virtual-dtor -Wnull-dereference -Wold-style-cast -Woverloaded-virtual -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-enum -Wtrampolines -Wundef -Wunused -Wuseless-cast -Wzero-as-null-pointer-constant")
else()
	message(FATAL_ERROR "Unknown compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_strict_flags}")

include(CTest)
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

find_package(fmt CONFIG REQUIRED)

find_package(Catch2 CONFIG REQUIRED)
include(Catch)

add_executable(test_logger test_logger.cpp test_fib.cpp catch_main.cpp logger.cpp)
target_link_libraries(test_logger PRIVATE Catch2::Catch2 fmt::fmt)
catch_discover_tests(test_logger)
