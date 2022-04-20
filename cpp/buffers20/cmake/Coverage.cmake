option(COV "Generate code coverage report" OFF)
if(COV)
	if(CMAKE_BUILD_TYPE STREQUAL Release)
		message(FATAL_ERROR "Release build is not suitable for coverage")
	endif()

	set(CMAKE_BUILD_TYPE Debug)

	message(STATUS "Generating code coverage report")

	include(cmake/CodeCoverage.cmake)
	append_coverage_compiler_flags()
	setup_target_for_coverage_lcov(
		NAME coverage
		EXECUTABLE ctest
		DEPENDENCIES test_buffers
		EXCLUDE "/usr/include/*" "${VCPKG_HOME}/*")
endif()
