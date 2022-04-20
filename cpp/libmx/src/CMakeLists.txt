set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

add_library(mx SHARED "")
target_sources(mx
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/mx.cpp
)
target_include_directories(mx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
set_target_properties(mx PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE 1
	SOVERSION ${PROJECT_VERSION}
	PUBLIC_HEADER "mx.hpp"
)
target_compile_options(mx PRIVATE ${COMMON_FLAGS})

install(
	TARGETS mx
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mx COMPONENT dev
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT bin
)

find_package(Boost 1.72 REQUIRED)
target_include_directories(mx SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(NOT CLANG_TIDY_EXE)
	message(WARNING "clang-tidy not found")
else()
    set_property(TARGET mx PROPERTY CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
	message(STATUS "Found ${CLANG_TIDY_EXE}")
endif()

find_program(CPPCHECK_EXE NAMES cppcheck)
if(NOT CPPCHECK_EXE)
    message(STATUS "cppcheck not found")
else()
    set_property(TARGET mx PROPERTY CXX_CPPCHECK ${CPPCHECK_EXE})
    message(STATUS "Found ${CPPCHECK_EXE}")
endif()
