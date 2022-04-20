set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# setup install dirs
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# compile_commands.json for varies tools
option(ENABLE_COMPILE_COMMAND "Enable compile_commands.json generation" ON)
if(ENABLE_COMPILE_COMMAND)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# enable ccache
option(ENABLE_CCACHE "Enable ccache" ON)
if(ENABLE_CCACHE)
  find_program(CCACHE_COMMAND NAMES ccache)
  if(CCACHE_COMMAND)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    message(STATUS "using ccache")
  else()
    message(WARNING "ccache not found")
  endif()
endif()

# add -flto
option(ENABLE_IPO "Enable link time optimization" ON)
if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)
  if(result)
    message(STATUS "IPO is supported")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "IPO is not supported: ${output}")
  endif()
endif()

# cppcheck
option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
if(ENABLE_CPPCHECK)
  find_program(CPPCHECK_COMMAND NAMES cppcheck)
  if(CPPCHECK_COMMAND)
    set(CMAKE_CXX_CPPCHECK ${CPPCHECK} --suppress=missingInclude --enable=all --inconclusive)
    message(STATUS "Found ${CPPCHECK_COMMAND}")
  else()
    message(SEND_ERROR "cppcheck requested but executable not found")
  endif()
endif()

# clang-tidy
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
if(ENABLE_CLANG_TIDY)
  find_program(CLANGTIDY_COMMAND NAMES clang-tidy clang-tidy-10)
  if(CLANGTIDY_COMMAND)
    set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY_COMMAND} -fix -extra-arg=-Wno-unknown-warning-option)
    message(STATUS "Found ${CLANGTIDY_COMMAND}")
  else()
    message(SEND_ERROR "clang-tidy requested but executable not found")
  endif()
endif()

# include-what-you-use
option(ENABLE_IWYU "Enable include-what-you-use" OFF)
if(ENABLE_IWYU)
  find_program(IWYU_COMMAND include-what-you-use)
  if(IWYU_COMMAND)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_COMMAND})
    message(STATUS "Found ${IWYU_COMMAND}")
  else()
    message(SEND_ERROR "include-what-you-use requested but executable not found")
  endif()
endif()

# valgrind
option(ENABLE_MEMCHECK_VALGRIND "Enable memcheck with valgrind" OFF)
if(ENABLE_MEMCHECK_VALGRIND)
  find_program(MEMORYCHECK_COMMAND NAMES valgrind)
  if(MEMORYCHECK_COMMAND)
    set(MEMORYCHECK_COMMAND_OPTIONS "--trace-
 origin=yes --leak-check=full --trace-children=full")
    add_custom_target(
      memcheck
      COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --test-action memcheck
      COMMAND cat "${CMAKE_BINARY_DIR}/Testing/Temporary/MemoryChecker.*.log")
    message(STATUS "Found ${MEMORYCHECK_COMMAND}")
  else()
    message(SEND_ERROR "valgrind requested but executable not found")
  endif()
endif()
