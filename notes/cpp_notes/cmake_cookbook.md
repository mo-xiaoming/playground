`cmake -H. -Bbuild`, `-H` where to find CMakeLists.txt, `-B` build directory

`cmake --build build` build target

`cmake --build build --target help` list all targets

  - `rebuild_cache`, incase new entries from the soruce need to be added

`cmake -DCMAKE_CXX_COMPILER=clang++` or `CXX=clang++ cmake ..`

`cmake --system-information`

`cmake -DCMAKE_CONFIGURATION_TYPES="Release;Debug` then by using `cmake --build build_release --config Release`

`cmake --build build -- VERBOSE=1`

```
set_target_properties(libmx PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
```

```
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	target_compile_definition(hello-world PUBLIC "IS_LINUX")
endif()

##ifdef IS_LINUX
##endif
```

`cmake -DCMAKE_PREFIX_PATH=<installation-prefix> ...`

`cmake --graphviz=build/r.dot build`
`dot -T png build/e.dot -o build/e.png`
