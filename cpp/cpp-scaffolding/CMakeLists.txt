cmake_minimum_required(VERSION 3.20)
project(CppTemplate LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)

include(cmake/Sanitizers.cmake)
include(cmake/StaticAnalyzers.cmake)

add_library(project_options INTERFACE)
include(cmake/CompilerOptions.cmake)
set_project_options(project_options)

add_library(project_warnings INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

add_library(dft_prj_cfg INTERFACE)
target_link_libraries(dft_prj_cfg INTERFACE project_options project_warnings)
enable_sanitizers(dft_prj_cfg)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup(TARGETS)

add_subdirectory(src)

#add_subdirectory(fuzzer)
