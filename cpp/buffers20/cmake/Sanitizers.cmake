option(SAN "Comma separated sanatizers" OFF)

if(SAN)
	if(CMAKE_BUILD_TYPE STREQUAL Release)
		message(FATAL_ERROR "Release build is not suitable for santizers")
	endif()

	set(CMAKE_BUILD_TYPE Debug)

	message(STATUS "Sanitizing with ${SAN}")

	if(SAN MATCHES "address")
		set(sanitizer_flags "-O1 -fsanitize=${SAN} -fno-optimize-sibling-calls -fsanitize-memory-track-origins")
	elseif(SAN MATCHES "thread")
		set(sanitizer_flags "-O2 -fsanitize=${SAN}")
	elseif(SAN MATCHES "memory")
		if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
			message(FATAL_ERROR "Only clang support memory sanitizer?, current compiler is ${CMAKE_CXX_COMPILER_ID}")
		endif()
		set(sanitizer_flags "-O1 -fsanitize=${SAN} -fno-optimize-sibling-calls -fsanitize-memory-track-origins")
	else()
		message(FATAL_ERROR "Unknown sanitizers ${SAN}")
	endif()
endif()
