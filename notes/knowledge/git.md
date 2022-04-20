## gpg
```bash
gpg --full-generate-key
gpg --list-secret-keys --keyid-format=long
gpg --armor --export 2B2FF1E29E07A36B
git checkout ignore_unreachable_rows
git config --global user.email '2188767+mo-xiaoming@users.noreply.github.com'
git config --global user.name 'Mo Xiaoming'
git config --global user.signingkey 2B2FF1E29E07A36B
if [ -r ~/.bash_profile ]; then echo 'export GPG_TTY=$(tty)' >> ~/.bash_profile;   else echo 'export GPG_TTY=$(tty)' >> ~/.profile; fi
#git config --global commit.gpgsign true
git commit --amend --signoff
git commit --amend --author='Mo Xiaoming <mo_xiao_ming@yahoo.com>' --no-edit
git push origin ignore_unreachable_rows -f
```

pull pull request `git pull origin pull/<ISSUE_ID>/head:<NEW_BRANCH_NAME>

show file content without checking out `git show <branch|commit>:file`

remove branch, locally `git branch -d localname`, remotely `git push origin --delete remotename`

list local git tags `git tag`, with description `git tag -n`

search with pattern `git tag -l <pattern>`, like `git tag -l v1*`

sort by version `git tag -l --sort=-version:refname <pattern>`, sort by commit date `git tag --sort=committerdate -l <pattern>`

list remote git tags `git ls-remote --tags <remote>`, like `git ls-remote --tags origin`

fetch remote tags `git fetch --all --tags`

check current tag `git describe -tags`

find root of project `git rev-parse --show-toplevel`

move master branch to three parents ahead `git branch -f master HEAD~3`

before C0 -> C1 -> C2 -> C3 -> C4 (master)

after C0 -> C1(master) -> C2 -> C3 -> C4
