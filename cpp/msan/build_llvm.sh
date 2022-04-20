set -eou pipefail

function build_clang() {
  local TY=clang-$1
  local SAN=${2:+""}
  CC=clang CXX=clang++ cmake -Bbuild-${TY} -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON LLVM_USE_SANITIZER:STRING="${SAN}" llvm -DCMAKE_INSTALL_PREFIX=$PWD/install-${TY} -GNinja -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt" -DLLVM_ENABLE_RTTI=ON
  cmake --build build-${TY}
  cmake --install build-${TY}
}

build_clang Debug
build_clang Release
build_clang ASanAndUBSan "Address;Undefined"
build_clang MSan         "MemoryWithOrigins"
build_clang TSan         "Thread"
