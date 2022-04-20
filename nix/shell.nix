{ pkgs ? import <nixpkgs> {} }:

let
  myVim = pkgs.callPackage ./vim.nix { inherit pkgs; };
in
  pkgs.mkShell {
    nativeBuildInputs = with pkgs; [ cmake ninja gitFull ccls ctags nodejs myVim ];
  }
