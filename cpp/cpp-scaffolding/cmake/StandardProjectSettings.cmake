# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Debug
        CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
    # Set the possible values of build type for cmake-gui, ccmake
    set_property(
        CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS
        "Debug"
        "Release"
        "MinSizeRel"
        "RelWithDebInfo")
endif()

option(ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" ON)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(ENABLE_IPO OFF)
endif()
if(ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(SEND_ERROR "IPO is not supported: ${output}")
    endif()
endif()

option(ENABLE_CCACHE "Enable ccache" OFF)
if(ENABLE_CCACHE)
  find_program(CCACHE_COMMAND NAMES ccache)
  if(CCACHE_COMMAND)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    message(STATUS "using ccache")
  else()
    message(WARNING "ccache not found")
  endif()
endif()

#if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
#    add_compile_options(-fcolor-diagnostics)
#elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#    add_compile_options(-fdiagnostics-color=always)
#else()
#    message(STATUS "No colored compiler diagnostic set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
#endif()
