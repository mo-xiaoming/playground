let pkgs = import <nixpkgs> {};
in pkgs.hello.override {
  stdenv = pkgs.llvmPackages_13.stdenv;
}
