cmake_minimum_required(VERSION 3.18)
if(NOT VCPKG_HOME)
	message(FATAL_ERROR "VCPKG_HOME is undefined")
endif()
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_HOME}/scripts/buildsystems/vcpkg.cmake")
project(Buffer20 VERSION 1.0.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CXXFlags.cmake)

include(cmake/Sanitizers.cmake)

include(cmake/Coverage.cmake)

include(cmake/BuildTypeGuard.cmake)

include(cmake/CCache.cmake)

include(cmake/Linters.cmake)

add_subdirectory(src)

#####################
#    tests
#####################
include(CTest)
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

add_subdirectory(tests)

include(cmake/Valgrind.cmake)
