git clone --depth=1 https://github.com/llvm/llvm-project.git llvm-12.0.1 -b llvmorg-12.0.1 --single-branch

lld nees zlib, which cause some problem under nix with llvm-8 and gcc10
cmake -Bbuild -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$PWD/install -GNinja llvm

cmake -Bbuild -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$PWD/install -DLLVM_USE_LINKER=lld -GNinja llvm

cmake -Bbuild -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$PWD/install -DLLVM_USE_LINKER=lld -DCLANG_ENABLE_STATIC_ANALYZER=OFF -DCLANG_ENABLE_ARCMT=OFF -DLLVM_BUILD_TOOLS=ON -DLLVM_BUILD_UTILS=OFF -DLLVM_ENABLE_PROJECTS=clang -GNinja llvm

#cmake -Bbuild -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$PWD/install -DLLVM_USE_LINKER=lld -DCLANG_ENABLE_STATIC_ANALYZER=OFF -DCLANG_ENABLE_ARCMT=OFF -DLLVM_BUILD_TOOLS=OFF -DLLVM_BUILD_UTILS=OFF -GNinja llvm

```bash
set -eou pipefail

LLVM_VER=12.0.1

git clone --depth=1 https://github.com/llvm/llvm-project.git llvm-${LLVM_VER} -b llvmorg-${LLVM_VER} --single-branch

cd llvm-${LLVM_VER}

function build_clang() {
  local TY=clang-$1
  local SAN=${2:+""}
  cmake -Bbuild-${TY} -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON LLVM_USE_SANITIZER:STRING="${SAN}" llvm -DCMAKE_INSTALL_PREFIX=$PWD/install-${TY} -GNinja -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  cmake --build build-${TY}
  cmake --install build-${TY}
}

build_clang Debug
build_clang Release
build_clang ASanAndUBSan "Address;Undefined"
build_clang MSan         "MemoryWithOrigins"
build_clang TSan         "Thread"

```

```
cmake -Bbuild-Debug        -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON                                               llvm -DCMAKE_INSTALL_PREFIX=$PWD/install-Debug        -GNinja -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt" && cmake --build build-Debug        && cmake --install build-Debug




cmake ../compiler-rt/ -DLLVM_CONFIG_PATH=$HOME/llvm-12.0.1/build/bin/llvm-config  -GNinja -DCMAKE_INSTALL_PREFIX=$HOME/llvm-12.0.1/install-ASanAndUBSan -CMAKE_BUILD_TYPE=Debug

cmake -Bbuild -S. -DLLVM_USE_LINKER=lld -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_PARALLEL_COMPILE_JOBS=1 -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_RTTI=ON llvm

cmake -Bbuild-rtti -S. -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS='clang' -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_INSTALL_PREFIX=$PWD/install-rtti llvm
cmake --build build-rtti

LLVM_DIR=/Volumes/DATA/projects/llvm-12.0.0/install-rtti/lib/cmake cmake -Bbuild -S. -DLLVM_TOOLS_BINARY_DIR=/Volumes/DATA/projects/llvm-12.0.0/install-rtti/bin

UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=halt_on_error=0 LSAN_OPTIONS=suppressions=supr.txt ./build/hobbes-test --tests Matching
```

```
// https://developers.redhat.com/blog/2021/05/05/memory-error-checking-in-c-and-c-comparing-sanitizers-and-valgrind

// clang/GCC option
-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment

// for lldb/gdb to prevent very short stack track and usually false leak detection
export ASAN_OPTIONS=abort_on_error=1:fast_unwind_on_malloc=0:detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1
```

```
// https://developers.redhat.com/blog/2021/04/23/valgrind-memcheck-different-ways-to-lose-your-memory

// to run tests
valgrind --quiet --errors-for-leak-kinds=all --show-leak-kinds=all --suppressions=local.supp --error-exitcode=99 --leak-check=full ./testprog
```

```
echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse
deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse
deb http://ddebs.ubuntu.com $(lsb_release -cs)-proposed main restricted universe multiverse" | \
sudo tee -a /etc/apt/sources.list.d/ddebs.list

sudo apt install ubuntu-dbgsym-keyring

sudo apt-get update

apt install debian-goodies

#find-dbgsym-packages [core_path|running_pid|binary_path]
```
