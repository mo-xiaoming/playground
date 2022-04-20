sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

git clone https://github.com/mo-xiaoming/playground
ln -s playground/notes/setup/_vimrc $HOME/.vimrc
ln -s playground/notes/setup/_screenrc $HOME/.screenrc

git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
vim +PluginInstall +qall
sudo apt install -y build-essential cmake vim-nox python3-dev
(cd ~/.vim/bundle/YouCompleteMe && python3 install.py --clangd-completer)

sudo apt install -y llvm-dev

sudo apt install -y libreadline-dev
git clone https://github.com/mo-xiaoming/hobbes
mkdir -p ~/.vim/syntax
cp hobbes/scripts/hobbes.vim ~/.vim/syntax
mkdir -p ~/.vim/ftdetect
cat <<-EOF>~/.vim/ftdetect/hobbes.vim
au BufRead,BufNewFile *.hob set filetype=hobbes
EOF

sudo passwd ubuntu

sudo sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
service ssh restart

