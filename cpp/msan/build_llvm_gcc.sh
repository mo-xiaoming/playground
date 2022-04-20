set -eou pipefail

function build_gcc() {
  local TY=gcc-$1
  local SAN=${2:+""}
  cmake -Bbuild-${TY} -S. -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_USE_SPLIT_DWARF=ON LLVM_USE_SANITIZER:STRING="${SAN}" llvm -DCMAKE_INSTALL_PREFIX=$PWD/install-${TY} -GNinja -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt" -DLLVM_ENABLE_RTTI=ON
  cmake --build build-${TY}
  cmake --install build-${TY}
}

build_gcc Debug
build_gcc Release
build_gcc ASanAndUBSan "Address;Undefined"
