final: prev: {
  llvm12WithDebug = prev.llvmPackages_12.llvm.overrideAttrs (oldAttrs: {
    doCheck = false;
    cmakeFlags = oldAttrs.cmakeFlags ++ [
      "-DCMAKE_BUILD_TYPE=Debug"
      "-DLLVM_BUILD_TESTS=OFF"
      "-DLLVM_ENABLE_FFI=OFF"
      "-DLLVM_ENABLE_RTTI=ON"
      "-DLLVM_TARGETS_TO_BUILD=X86"
      "-DLLVM_BUILD_TOOLS=OFF"
      "-DLLVM_ENABLE_ZLIB=OFF"
      "-DLLVM_INCLUDE_BENCHMARKS=OFF"
      "-DLLVM_INCLUDE_TESTS=OFF"
      "-DLLVM_INCLUDE_TOOLS=OFF"
      "-DLLVM_OPTIMIZED_TABLEGEN=ON"
    ];
  });
}
