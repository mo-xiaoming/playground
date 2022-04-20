{ pkgs ? import <nixpkgs> {} }:
let
  ghc = pkgs.haskellPackages.ghcWithPackages (ps: with ps; [
    haskell-language-server

    cabal2nix
    cabal-install

    QuickCheck quickcheck-script
    HUnit
    doctest
  ]);
in
  pkgs.mkShell {
    nativeBuildInputs = with pkgs; [ inotifyTools ];

    buildInputs = [ ghc ];

    shellHook = "eval $(egrep ^export ${ghc}/bin/ghc)";
  }
