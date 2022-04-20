#!/usr/bin/env bash

set -e

TOP_BUILD_DIR=build
[[ -d $TOP_BUILD_DIR ]] || mkdir -p "$TOP_BUILD_DIR"

if (($# != 0)); then
	export CC=gcc
	export CXX=g++

	cmake -S. -B$TOP_BUILD_DIR/release_gcc -DVCPKG_HOME=$HOME/projects/vcpkg
	cmake --build $TOP_BUILD_DIR/release_gcc -j4
	cmake --build $TOP_BUILD_DIR/release_gcc -t memcheck || exit 1
	cmake --build $TOP_BUILD_DIR/release_gcc -t threadcheck || exit 1

	cmake -S. -B$TOP_BUILD_DIR/address_gcc -DSAN="address,undefined" -DVCPKG_HOME=$HOME/projects/vcpkg
	cmake --build $TOP_BUILD_DIR/address_gcc -j4
	cmake --build $TOP_BUILD_DIR/address_gcc -t test

	cmake -S. -B$TOP_BUILD_DIR/thread_gcc -DSAN="thread" -DVCPKG_HOME=$HOME/projects/vcpkg
	cmake --build $TOP_BUILD_DIR/thread_gcc -j4
	cmake --build $TOP_BUILD_DIR/thread_gcc -t test

	cmake -S. -B$TOP_BUILD_DIR/coverage_gcc -DCOV:BOOL=ON -DVCPKG_HOME=$HOME/projects/vcpkg
	cmake --build $TOP_BUILD_DIR/coverage_gcc -j4
	cmake --build $TOP_BUILD_DIR/coverage_gcc -t coverage

	export CC=clang
	export CXX=clang++

	cmake -S. -B$TOP_BUILD_DIR/address_clang -DSAN="address,undefined" -DVCPKG_HOME=$HOME/projects/vcpkg
	cmake --build $TOP_BUILD_DIR/address_clang -j4
	cmake --build $TOP_BUILD_DIR/address_clang -t test

	cmake -S. -B$TOP_BUILD_DIR/thread_clang -DSAN="thread" -DVCPKG_HOME=$HOME/projects/vcpkg
	cmake --build $TOP_BUILD_DIR/thread_clang -j4
	cmake --build $TOP_BUILD_DIR/thread_clang -t test
fi

export CC=clang
export CXX=clang++

cmake -S. -B$TOP_BUILD_DIR/release_clang -DVCPKG_HOME=$HOME/projects/vcpkg
cmake --build $TOP_BUILD_DIR/release_clang -j4
cmake --build $TOP_BUILD_DIR/release_clang -t test

ln -sf $TOP_BUILD_DIR/release_clang/compile_commands.json
