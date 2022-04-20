if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Release)
	message(STATUS "Release build")
elseif(CMAKE_BUILD_TYPE STREQUAL Debug)
	message(STATUS "Debug build")
else()
	message(FATAL_ERROR "Unknown build type ${CMAKE_BUILD_TYPE}")
endif()

