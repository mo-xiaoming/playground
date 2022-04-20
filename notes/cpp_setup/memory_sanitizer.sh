#!/usr/bin/env bash

set -e

#pushd .
## clone LLVM
#git clone --depth=1 https://github.com/llvm/llvm-project
#cd llvm-project
#[[ -d build ]] || mkdir build
#cd build
## configure cmake
#cmake -GNinja ../llvm \
#	-DCMAKE_BUILD_TYPE=Release \
#	-DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" \
#	-DCMAKE_C_COMPILER=clang \
#	-DCMAKE_CXX_COMPILER=clang++ \
#	-DLLVM_USE_SANITIZER=MemoryWithOrigins
#
## build the libraries
#cmake --build . -- cxx cxxabi
#LLVM_BUILD=$PWD
#popd
#
#
#pushd .
#git clone https://github.com/google/googletest
[[ -d gtest-msan ]] || mkdir gtest-msan
cd gtest-msan
MSAN_CFLAGS="-fsanitize=memory -stdlib=libc++ -I${LLVM_BUILD}/include -I${LLVM_BUILD}/include/c++/v1"
MSAN_LINKER_FLAGS="-fsanitize=memory -stdlib=libc++ -L${LLVM_BUILD}/lib -lc++abi"
cmake ../googletest -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS="$MSAN_CFLAGS" -DCMAKE_CXX_FLAGS="$MSAN_CFLAGS" -DCMAKE_EXE_LINKER_FLAGS="$MSAN_LINKER_FLAGS -DCMAKE_STATIC_LINKER_FLAGS=$MSAN_LINKER_FLAGS"

make -j4
#popd
