{ pkgs ? import <nixpkgs> {}, }:
rec {
  myProject = pkgs.stdenv.mkDerivation {
    pname = "hi";
    version = "dev-0.1";
    buildInput = with pkgs; [ gbenchmark (callPackage ./catch2.nix {}) ];
  };
}
