#!/bin/sh -e

apt install -y git

# install clang10
sudo apt-get install clang-10 clang-tools-10 clang-10-doc libclang-common-10-dev libclang-10-dev libclang1-10 clang-format-10 python3-clang-10 clangd-10 libfuzzer-10-dev lldb-10 lld-10 libc++-10-dev libc++abi-10-dev libomp-10-dev clang-tidy-10
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 60 \
	--slave /usr/bin/clang++ clang++ /usr/bin/clang++-10 \
	--slave /usr/bin/clangd clangd /usr/bin/clangd-10 \
	--slave /usr/bin/clang-format clang-format /usr/bin/clang-format-10 \
	--slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-10

# install gcc9
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
#sudo apt-get -y install g++-9
#sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo apt install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9 gcc-10 g++-10 -y

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10

sudo update--alternatives --config gcc

# setup Vundle.vim
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
(export NOTESDIR=$PWD && cd && ln -s .vimrc ${NOTESDIR}/_vimrc)

# install ycm
git clone https://github.com/ycm-core/YouCompleteMe ~/.vim/bundle/YouCompleteMe
sudo apt -y install python3-dev
(cd ~/.vim/bundle/YouCompleteMe && git submodule update --init --recursive && python3 install.py --clangd-completer)

#curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
#sudo apt install nodejs
