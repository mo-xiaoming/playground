```bash
ssh-keygen -t rsa -b 4096 -C 'skelixos@gmail.com'
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
# copy/paste ~/.ssh/id_rsa.pub to github
```

```bash
git init
git config user.email 'mo_xiao_ming@yahoo.com'
git config user.name 'Xiaoming Mo'
git add .
git commit -m 'INIT'
git remote add origin git@github.com:mo-xiaoming/notes.git
git push -u origin master
```
