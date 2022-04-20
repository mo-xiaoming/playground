#!/usr/bin/env bash

set -e

exec > /dev/null

pushd . 2>/dev/null
[[ -d llvm-project ]] || git clone --depth=1 https://github.com/llvm/llvm-project
cd llvm-project/
git pull
[[ -d build ]] || mkdir build
cd build
cmake -GNinja ../llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_SANITIZER=MemoryWithOrigins && cmake --build . -- cxx cxxabi
LLVM_BUILD=$PWD
popd 2>/dev/null

LLVM_BUILD=$PWD/llvm-project/build

pushd . 2>/dev/null
[[ -d googletest ]] || git clone https://github.com/google/googletest
(cd googletest && git pull)
sed -i "s/ -Werror//g" googletest/googletest/cmake/internal_utils.cmake
[[ -d gtest-msan ]] || mkdir gtest-msan
cd gtest-msan
MSAN_CFLAGS="-fsanitize=memory -stdlib=libc++ -nostdinc++ -fno-omit-frame-pointer -L$LLVM_BUILD/lib -lc++abi -I$LLVM_BUILD/include -I$LLVM_BUILD/include/c++/v1"
cmake ../googletest -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS="$MSAN_CFLAGS" -DCMAKE_CXX_FLAGS="$MSAN_CFLAGS"
make -j12
popd 2>/dev/null

echo clang++ ${MSAN_CFLAGS} -g test.cc -Igoogletest/googletest/include/ -Lgtest-msan/lib/ -lpthread -lgtestd -lgtest_maind -Wl,-rpath,$LLVM_BUILD/lib >&2
clang++ ${MSAN_CFLAGS} -g test.cc -Igoogletest/googletest/include/ -Lgtest-msan/lib/ -lpthread -lgtestd -lgtest_maind -Wl,-rpath,$LLVM_BUILD/lib
