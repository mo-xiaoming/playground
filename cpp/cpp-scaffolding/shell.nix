{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  nativeBuildInputs = builtins.attrValues { inherit (pkgs) cmake ninja; inherit (pkgs.llvmPackages_12) libclang; };
  buildInputs = builtins.attrValues { inherit (pkgs) spdlog catch2; };
}
