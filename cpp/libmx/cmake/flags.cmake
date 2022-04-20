macro(define_cxx_flags)
	if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
		list(APPEND COMMON_FLAGS 
			-g
			-O3
			-fno-omit-frame-pointer
			-fPIC
			-pie
			-march=native
			-mtune=native
			-flto
			-Wall
			-Wextra
			-Wpedantic
			-Wcast-align=strict
			-Wcast-qual
			-Wconversion
			-Wdouble-promotion
			-Wduplicated-branches
			-Wduplicated-cond
			-Wfloat-equal
			-Wformat=2
			-Wlogical-op
			-Wmisleading-indentation
			-Wmissing-declarations
			-Wmissing-include-dirs
			-Wnon-virtual-dtor
			-Wnull-dereference
			-Wold-style-cast
			-Woverloaded-virtual
			-Wpointer-arith
			-Wredundant-decls
			-Wshadow
			-Wsign-conversion
			-Wswitch-enum
			-Wtrampolines
			-Wundef
			-Wunused
			-Wuseless-cast
			-Wzero-as-null-pointer-constant
		)
	elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		list(APPEND COMMON_FLAGS
			-g
			-O3
			-fno-omit-frame-pointer
			-fno-rtti
			-fPIC
			-march=native
			-mtune=native
			-Weverything
			-Wno-c++98-compat
			-Wno-c++98-compat-pedantic
			-Wno-c99-extensions
			-pedantic
			-pedantic-errors
		)
	elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		list(APPEND COMMON_FLAGS
			/W4
			/Wall
			/WX
		)
	endif()
endmacro()
