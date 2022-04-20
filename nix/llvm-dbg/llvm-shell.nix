{ pkgs ? import <nixpkgs> {}, llvmVersion ? "12.0.1" }:
pkgs.mkShell rec {
  nativeBuildInputs = with pkgs; [ cmake ninja python2 gdb valgrind ];
  buildInputs = with pkgs; [ zlib ncurses readline libxml2 ];
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
    export LLVM_DIR="$HOME/llvm/${llvmVersion}/install/lib/cmake/llvm"
    export CMAKE_EXPORT_COMPILE_COMMANDS=ON
  '';
}
