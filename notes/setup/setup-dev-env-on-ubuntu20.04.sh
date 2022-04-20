#!/usr/bin/env sh

#ssh-keygen -t rsa -b 4096 -C "skelixos@gmail.com"
#eval "$(ssh-agent -s)"
#ssh-add ~/.ssh/id_rsa
#cat ~/.ssh/id_rsa.pub

sudo apt-get install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9

sudo apt-get install -y clang clang-format clang-tidy \
	python3 python3-pip python3-venv \
	git \
	ripgrep \
	default-jdk \
	vim-nox \
	ccache \
	curl \
	lcov

git config --global user.name "Mo Xiaoming"
git config --global user.email "mo_xiao_ming@yahoo.com"
git config --global core.editor vim

(mkdir ~/projects ; cd ~/projects &&
git clone git@github.com:mo-xiaoming/notes &&
git clone git@github.com:mo-xiaoming/cpp-presentation.git &&
git clone git@github.com:mo-xiaoming/libmx)

cd && \
	ln -sf ~/projects/notes/setup/_vimrc .vimrc && \
	ln -sf ~/projects/notes/setup/_screenrc .screenrc && \
	ln -sf ~/projects/notes/setup/_pip .pip

pip3 install --user --upgrade cmake conan ninja
export PATH=~/.local/bin:$PATH

git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim && \
	git clone https://github.com/ycm-core/YouCompleteMe.git ~/.vim/bundle/YouCompleteMe && \
	cd ~/.vim/bundle/YouCompleteMe && git submodule update --init --recursive && \
	vim +PluginInstall +qall && \
	python3 install.py --clangd-completer --java-completer

#virtualbox
sudo usermod -G vboxsf -a $(whoami)
newgrp vboxsf

#vmware fusion
#.host:/ /mnt/hgfs fuse.vmhgfs-fuse defaults,allow_other 0 0
#vmhgfs-fuse /mnt/hgfs fuse defaults,allow_other 0 0
