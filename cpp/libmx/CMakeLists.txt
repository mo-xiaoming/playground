cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project(MyLib VERSION 1.0.0.0 LANGUAGES CXX)

configure_file(version.h.in generated/version.h @ONLY)

# CXX flags
include(cmake/flags.cmake)
define_cxx_flags()

# compile_commands.json for varies tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# setup install dirs
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# lib
add_subdirectory(src)

# tests
enable_testing()
add_subdirectory(tests)

# extern tools
include(CTest)

find_program(MEMORYCHECK_COMMAND NAMES valgrind)
if(NOT MEMORYCHECK_COMMAND)
	message(WARNING "valgrind not found")
else()
	set(MEMORYCHECK_COMMAND_OPTIONS "--trace-origin=yes --leak-check=full --trace-children=full")
	add_custom_target(memcheck
		COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --test-action memcheck
		COMMAND cat "${CMAKE_BINARY_DIR}/Testing/Temporary/MemoryChecker.*.log")
	message(STATUS "Found ${MEMORYCHECK_COMMAND}")
endif()

