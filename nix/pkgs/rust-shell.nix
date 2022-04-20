{ pkgs ? import <nixpkgs> {} }:
with pkgs; mkShell {
  nativeBuildInputs = [
    rustc cargo
    gcc
  ];
  buildInputs = [ rustfmt clippy ];

  RUST_SRC_PATH = "${rust.packages.stable.rustPlatform.rustLibSrc}";
}
