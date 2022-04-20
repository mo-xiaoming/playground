cmake_minimum_required(VERSION 3.17)
project(danilafe)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  message(STATUS "IPO/LTO enabed")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_Release TRUE)
else()
  message(SEND_ERROR "IPO/LTO not supported: ${output}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "")

add_compile_options(-fno-rtti)
add_link_options(-fno-rtti)

add_compile_options("$<$<CONFIG:Release>:-Werror;-Wfatal-errors>")
add_compile_options(-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wfloat-equal -Wmissing-declarations -Wmissing-include-dirs -Wredundant-decls -Wswitch-enum -Wundef -Wzero-as-null-pointer-constant)
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wmisleading-indentation;-Wduplicated-cond;-Wduplicated-branches;-Wlogical-op;-Wuseless-cast>")

add_compile_options("$<$<NOT:$<CONFIG:Release>>:-O0;-ggdb3;-fno-omit-frame-pointer;-fno-inline;-fno-sanitize-recover=all>")

add_compile_options("$<$<CONFIG:ASanAndUBSan>:-fsanitize=address,undefined;-fno-optimize-sibling-calls;-fsanitize=float-divide-by-zero;-fsanitize=float-cast-overflow;-U_FORTIFY_SOURCE>")
add_link_options("$<$<CONFIG:ASanAndUBSan>:-fsanitize=address,undefined;-fno-optimize-sibling-calls;-fsanitize=float-divide-by-zero;-fsanitize=float-cast-overflow>")
add_compile_options("$<$<CXX_COMPILER_ID:AppleClang,Clang>:-fsanitize=local-bounds,float-divide-by-zero,implicit-conversion,nullability,integer>")
add_link_options("$<$<CXX_COMPILER_ID:AppleClang,Clang>:-fsanitize=local-bounds,float-divide-by-zero,implicit-conversion,nullability,integer>")

find_package(Catch2 REQUIRED)
enable_testing()
include(CTest)
include(Catch)

add_subdirectory(src/lib)
