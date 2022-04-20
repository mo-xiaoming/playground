{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [ ghc gnupg cacert nix mpv iputils debianutils file git curl firefox lesspipe less tmux screen ripgrep clang_12 ctags nodejs (callPackage ~/playground/nix/vim.nix { inherit pkgs; useNeovim=true; }) ];
}
