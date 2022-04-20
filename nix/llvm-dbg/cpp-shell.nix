let
  pkgs = import <nixpkgs> {};
  addClangTool = oldAttrs: {
    nativeBuildInputs = oldAttrs.nativeBuildInputs ++ [ pkgs.clang-tools ];
    CPATH = pkgs.lib.makeSearchPathOutput "dev" "include" oldAttrs.buildInputs;
  };
in
  (import ./default.nix {}).hobbes.overrideAttrs addClangTool
