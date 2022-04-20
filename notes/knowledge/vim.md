##### changes
`:changes` change list
`g;` and `g,` jump to the location of the changes
`gi` last change location

##### split
`^w h` change splits to horizontal
`^w k` chagne splits to vertical

##### man
`\K` man page on word under the cursor, or `Man command`

##### normal command
`1,3normal A;` append `;` to first 3 lines
`<,>normal farc` replace first `a` with `c` on each line in selection

##### rever buffer
`:edit!`

##### reload .vimrc
`:so $MYVIMRC`

##### netrw
`:Explore` `:Sexplore` `:Vexplore` open `netrw` in current/horizontal/vertical window
`i` toggle view types, `let g:netrw_liststyle=3`
`I` toggle banner, `let g:netrw_banner=0`
`1` `2` `3` `4` open files in horizontal split/vertical split/new tab/prev window, `let g:netrw_browse_split = 1`
`let g:netrw_winsize=25` 25% of the screen width

##### open last file at last position
`^o ^o ^o...`

##### edit remote file
`vim scp://user@myserver[:port]//path/to/file.txt`

##### edit compressed file
`vim archive.tar.gz`

#### VERBS_MODIFIERS_NOUNS

##### VERBS
`d` delete
`c` change
`y` yank
`v` visually select

##### MODIFIERS
`i` inside
`a` around
`NUM` number(1, 2, 10)
`t` searches for something and **stops before** it
`f` searches for something and **lands on** it
`/` find a string

##### NOUNS
`s` sentence
`)` sentence
`p` paragraph
`}` paragraph
`t` tag (think HTML/XML)
`b` block (think programming)

##### VERBS_MODIFIERS_NOUNS examples
`d2w` delete two words
`cis` change inside sentence
`yip` yank inside paragraph
`ct<` change to open bracket
`v2i{` select everything inside the second tier braces

`H` move to the top of the screen
`M` move to the middle of the screen
`L` move to the bottom of the screen

`^U` move up half a screen
`^D` move down half a screen
`^E` scroll up one line
`^Y` scroll down one line

`I` insert at the beginning of the line
`A` append at the end of the line

`^i`/`^o` jump back and forth between two places

`^R` redo

`qa` start recording a macro named "a"
`q` stop recording
`@a` play back the macro

`gd` goto definition
`gf` open file under cursor

<C-X><C-L> line completion

`guu` lower line
`gUU` upper line

`=` reindent
`=%` reindent current braces
`G=gg` reindent file

`:e!` return to unmodifed file

`''` last cursor spot
``\`.`` last modified spot

CTRL_W HJKL move win position
`3fx`表示移动到光标右边的第3个'x'字符上。
`;`命令重复前一次输入的`f`, `t`, `F`, `T`命令，而`,`命令会反方向重复前一次输入的`f`, `t`, `F`, `T`命令。这两个命令前也可以使用数字来表示倍数。

`*`    转到当前光标所指的单词下一次出现的地方
`#`    转到当前光标所指的单词上一次出现的地方

`:set spell spelllang=en_us` spell checking
`:set spell`/`:set nospell` enable/disable spell checking

```
syntax on
filetype plugin indent on

set hlsearch
set incsearch
set ignorecase
set smartcase

set paste

set background=light

set switchbuf=useopen

set showcmd

set wildmenu

set shiftwidth=4
set tabstop=4
set expandtab

autocmd FileType python set makeprg=pylint\ --reports=n\ --output-format=parseable\ %

map <F5> :cprevious<CR>
map <F6> :cnext<CR>
map <F7> :clist<CR>
```
