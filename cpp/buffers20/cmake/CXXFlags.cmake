set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -lto
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(cxx_base_flags "-Wall -Wextra -Wshadow -march=native -mtune=native")
set(cxx_werror_flags "-Werror")
set(cxx_no_rtti_flags "-fno-rtti")
set(cxx_no_exception_flags "-fno-exceptions")
set(cxx_debug_info_flags "-g")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(cxx_strict_flags "-Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-c99-extensions -Wno-gnu-zero-variadic-macro-arguments -Wno-global-constructors")
elseif(CMAKE_COMPILER_IS_GNUCXX)
	set(cxx_strict_flags "-Wcast-align=strict -Wcast-qual -Wconversion -Wdouble-promotion -Wduplicated-branches -Wduplicated-cond -Wfloat-equal -Wformat=2 -Wlogical-op -Wmisleading-indentation -Wmissing-declarations -Wmissing-include-dirs -Wnon-virtual-dtor -Wnull-dereference -Wold-style-cast -Woverloaded-virtual -Wpointer-arith -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-enum -Wtrampolines -Wundef -Wunused -Wuseless-cast -Wzero-as-null-pointer-constant")
else()
	message(FATAL_ERROR "Unknown compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

set(CMAKE_CXX_FLAGS "${cxx_debug_info_flags} ${cxx_base_flags} ${cxx_no_rtti_flags} ${cxx_strict_flags}")

set(CMAKE_CXX_FLAGS_DEBUG "-fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
