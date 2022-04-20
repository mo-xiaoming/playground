#### run once gdbserver

1. on taret

```
$ gdbserver localhost:2000 a.out
```

2. on host
```
$ gdb a.out
(gdb) target remote 192.168.20.11:2000
```

#### attach to a running process on target

1. on host 
```
(gdb) attach 3850
```

#### run gdbserver in multi-process mode

1. on target
```
$ gdbserver --multi localhost:2000
```

2. on host 
```
$ gdb
(gdb) set remote exec-file /a.out # set the program which you want to debug in tthe target
(gdb) file /a.out # load the debugging symbols from the program in the host
(gdb) target extended-remote 192.168.20.11:2000
```

#### shutown multi-process gdbserver
1. on host
```
(gdb) monitor exit
```

#### get pid
```
$ ps -C ipwenum -o pid h
```

#### gdb how-to change source file dir

```
(gdb) list
source.c: No such file or directory
(gdb) directory /some/path/to/src 
Source directories searched: /some/path/to/src:$cdir:$cwd
(gdb) list
.
.
.
```
`directory` add a path to beginning of search path for srouce files.

```
(gdb) list
source.c: No such file or directory
(gdb) set substitute-path /FROM/path /TO/path
(gdb) list
.
.
.
```
`set substitute-path FROM TO` adds a substitution rule replacing FROM into TO in source file names.
