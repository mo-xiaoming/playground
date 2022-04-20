##### List all files in an RPM package

`rpm -qlpv some.rpm`

`-q` specifies it as a query command

`-l` lists files in the package

`-p` quries the uninstalled package file

`-v` verbose

##### Extract files from an RPM package

`rpm2cpio some.rpm |cpio -idmv`

`-i` extracts the files from the archive

`-d` create the leading directories where needed

`-m` preserve the file modification time

`-v` verbose


##### Show preinstall and postinstall scripts

`rpm -qp --scripts some.rpm`


#### show rpm info

`rpm -qip some.rpm`

#### show installed rpm info

`rpm -qi some.rpm`

#### check dependency

`rpm -qpR some.rpm`

#### check an installed rpm

`rpm -q some`

#### show all installed rpm

`rpm -qa`


#### query a file belongs to which rpm

`rpm -qf /usr/bin/ls`
