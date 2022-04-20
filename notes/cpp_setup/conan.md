## first installation

```
$ conan profile new default --detect  # Generates default profile detecting GCC and sets old ABI
$ conan profile update settings.compiler.libcxx=libstdc++11 default  # Sets libcxx to C++11 ABI
```

## [repo](https://conanio.center)

## update

`pip install conan --upgrade`

## search

`conan search poco -r conan-center`

## inspect

`conan insepct poco/1.9.4 [-r conan-center]`

### basic conanfile.txt and cmake
```
[requires]
poco/1.9.4

[generators]
cmake
```
```
mkdir build && cd build
conan install .. --build=missing
```

```
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(md5 md5.cpp)
target_link_libraries(md5 ${CONAN_LIBS})
```

## existing packages `conan search "*"`

## project's denendencies `conan info .`, in graph `conan info . --graph=file.html`

## options `conan inspect poco/1.9.4 -a=options` / `conan inspect poco/1.9.4 -a=default_options`

add another section to `conanfile.txt`

```
[options]
poco:shared=True
```

or `conan install .. -o poco:shared=True -o openssl:shared=True` or `conan install . -o *:shared=True`

## copy files

```
[imports]
bin, *.dll -> ./bin # copyies all dll files from packages bin folder to my "bin" folder
```

## multi configurations

```
conan install md5 -s build_type=Debug -if md5_build_debug -b missing
conan install md5 -s build_type=Release -if md5_build_Release -b missing

(cd md5_build_debug && cmake ../md5 -GNinja && cmake --build . --config Debug)
(cd md5_build_release && cmake ../md5 -GNinja && cmake --build . --config Release)
```

## with clion

```
conan install . -s build_type=Debug --install-folder=cmake-build-debug
conan install . -s build_type=Release --install-folder=cmake-build-release
```

## multi configuration

```
conan install . -g cmake_find_package_multi -s build_type=Debug
conan install . -g cmake_find_package_multi -s build_type=Release
```

must be `find_package(xxx CONFIG)` in cmake file to find `xxxConfig.cmake` instead of `Findxxx.cmake`

`CONAN_COLOR_DARK=1` for light background
