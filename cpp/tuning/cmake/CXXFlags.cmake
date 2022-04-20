  option(ENABLE_WERRORS "Treat compiler warnings as errors" TRUE)
  option(ENABLE_DEBUG_SYMBOLS "With debug symbols" TRUE)

  list(
    APPEND
    _COMMON_FLAGS
    -march=native
    -mtune=native
    -Wall
    -Wextra
    -pedantic
    -pedantic-errors
    -Wshadow
	-ffast-math
    -Wcast-align # cast requires alignment of the target is increased
    -Wcast-qual # pointer casted to as to remove a type quanlifier
    -Wconversion # implicit conversions that may alter a value
    -Wdouble-promotion # implicitly promote 'float' to 'double', on some platform might be expensive
    -Wfloat-equal # floating-point values are used in equality comparisons
    -Wformat=2 # check for printf/scanf
    -Wmisleading-indentation # indentation does not reflect the block structure
    -Wmissing-include-dirs # if user-supplied include directory does not exist
    -Wnon-virtual-dtor # class has virtual functions but non virtual diestructor
    -Wnull-dereference # detect null pointer dereferencing
    -Wold-style-cast # C-style cast to a non-void type
    -Woverloaded-virtual # functions hides virtual functions from base class
    -Wredundant-decls # anything is declared more than once in the same scope
    -Wsign-conversion # conversions between signed and unsigned
    -Wswitch-enum # if a enum case is missing
    -Wundef # if undefined identifiers is evaluated in an "#if"
    -Wunused # all unused something combined
    -Wzero-as-null-pointer-constant # literal 0 is used as null pointer constant
  )

  if(ENABLE_WERRORS)
    list(APPEND _COMMON_FLAGS -Werror)
  endif()

  if(ENABLE_DEBUG_SYMBOLS)
    list(APPEND _COMMON_FLAGS -g)
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(
      APPEND
      COMMON_FLAGS
      ${_COMMON_FLAGS}
      -Wduplicated-branches # an if-else has identical branches
      -Wduplicated-cond # duplicated conditions in an if-else-if chain
      -Wlogical-op # suspicious uses of logical operators
      -Wmissing-declarations # if a global function is defined without a previous declaration
      -Wuseless-cast # cast to its own type
    )
  endif()
