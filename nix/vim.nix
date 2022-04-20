{ pkgs, useNeovim ? true }:

let
  vimConfig = {
    customRC = builtins.readFile ./vimrc;
    packages.myVimPackage = with pkgs.vimPlugins; {
      # loaded on launch
      start = [
        coc-nvim
        coc-clangd
        coc-explorer
        coc-json
        rainbow
        vim-lsp-cxx-highlight
        csv-vim
        vim-airline
        vim-gitgutter
        vim-nix
        #git-blame-nvim
      ];
      # manually loadable by calling `:packadd $plugin-name`
      opt = [ /* ... */ ];
      # To automatically load a plugin when opening a filetype, add vimrc lines
      # like:
      # autocmd FileType php :packadd phpCompletion
    };
  };
in

if useNeovim then
  pkgs.neovim.override {
    viAlias = true;
    vimAlias = true;
    withNodeJs = true;
    withPython3 = true;
    configure = vimConfig;
  }
else
  pkgs.vim_configurable.customize {
    name = "vim";
    vimrcConfig = vimConfig;
  }
