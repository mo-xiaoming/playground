```bash
first$ gpg --list-secret-keys my@email.com

sec   rsa4096 2021-06-15 [SC]
      ABCDEFG #<- some hash ->
uid           [ultimate] Mo Xiaoming (yahoo@inno.ms) <my@email.com>
ssb   rsa4096 2021-06-15 [E]

first$ gpg --export-secret-keys <- some hash -> > pri.key

second$ gpg --import pri.key
```
