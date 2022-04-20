cmake_minimum_required(VERSION 3.16)
project(TestGTest VERSION 1.0.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(cxx_compiler_is_clang TRUE)
elseif (CMAKE_COMPILER_IS_GNUCXX)
    set(cxx_compiler_is_gcc TRUE)
else ()
    message(FATAL_ERROR "unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif ()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-rtti -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-optimize-sibling-calls -fno-omit-frame-pointer")

macro(add_sanitizer name flags)
    if (NOT CMAKE_BUILD_TYPE STREQUAL Debug)
        message(WARNING "sanitizer is ${name}, but build type is ${CMAKE_BUILD_TYPE}, set to Debug")
    endif ()
    set(CMAKE_BUILD_TYPE Debug)
    if (cxx_compiler_is_clang)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${flags}")
    else ()
        message(FATAL_ERROR "unsupported compiler ${cxx_compiler_is_clang} for sanitizer ${name}")
    endif ()
endmacro()

option(SANITIZE_ADDRESS "enable address santitizer" OFF)
if (SANITIZE_ADDRESS)
	add_sanitizer("address" "-O1 -fsanitize=address -fsanitize-address-use-after-scope")
endif ()

option(SANITIZE_UNDEFINED "enable undefined santitizer" OFF)
if (SANITIZE_UNDEFINED)
	add_sanitizer("undefined" "-O1 -fsanitize=undefined,float-divide-by-zero,unsigned-integer-overflow,implicit-conversion,local-bounds,implicit-integer-truncation,implicit-integer-arithmetic-value-change,integer,nullability")
endif ()

if (cxx_compiler_is_clang)
    set(cxx_strict_flags "-Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-exit-time-destructors -Wno-global-constructors -Wno-gnu-zero-variadic-macro-arguments")
elseif (cxx_compiler_is_gcc)
    set(cxx_strict_flags "-Wcast-align=strict -Wcast-qual -Wconversion -Wdouble-promotion -Wduplicated-branches -Wduplicated-cond -Wfloat-equal -Wformat=2 -Wlogical-op -Wmisleading-indentation -Wmissing-declarations -Wmissing-include-dirs -Wnon-virtual-dtor -Wnull-dereference -Wold-style-cast -Woverloaded-virtual -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-enum -Wtrampolines -Wundef -Wunused -Wuseless-cast -Wzero-as-null-pointer-constant")
else ()
    message(FATAL_ERROR "unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_strict_flags}")

mark_as_advanced(cxx_strict_flags cxx_compiler_is_gcc cxx_compiler_is_clang)


include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

find_package(spdlog REQUIRED)

enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)
add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock spdlog::spdlog)
gtest_discover_tests(tests)
