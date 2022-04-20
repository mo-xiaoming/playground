# GCC Options

For improved debugging experience, consider setting the optimizaiton with the `-Og` option.

To include also macro definitions in the debug information, use the `-g3` option instead of `-g`

The `-fcompare-debug` option tests code compiled by GCC with debug information and without debug information

For release builds, use the optimization options `-O2`

## Release version options

`gcc ... -O2 -g -Wall -Wl,-z,now,-z,relro -fstack-protector-strong -fstack-clash-protection -D_FORTIFY_SOURCE=2 ...`

   * For programs, add the `-fPIE` and `-pie` Position Independent Executable options
   * For dynamically linked libraries, the mandotory `-fPIE` (Position Independent Code) option indirectly increases security

## Development options

*Use these options in conjunction with the options for the release version*

`gcc ... -Walloc-zero -Walloca-larger-than -Wextra -Wformat-security -Wvla-larger-than`

*These options are C related*

## Use `rpath` or `LD_LIBRARY_PATH` to specify the shared lib path

`gcc ... -Llibrary_path -lfoo -Wl,-rpath=library_path ...`

or

`LD_LIBRARY_PATH=library_path:$LD_LIBRARY_PATH ./program`

## Shared lib related resources

`man ld.so`

`cat /etc/ld.so.conf`

`ldconfig -v`

## Creating dynamic libraries

`gcc ... -c -fPIC some_file.c ...`

`gcc -shared -o libfoo.so.x.y -Wl,-soname,libfoo.so.x some_file.o ...`

`cp libfoo.so.x.y /usr/lib64`

`ln -s libfoo.so.x.y libfoo.so.x`

`ln -s libfoo.so.x libfoo.so`

## Get depending shared libs

`objdump -p program |grep -e '(SONAME|NEEDED)'`

## Debugging

`set no-stop on` and `set target-async on` make GDB stop only the thread that is examined

`strace -fvttTyy -s 256 -e trace=all ./program`

   * `-f` trace child processes or threads
   * `-v` verbose
   * `-tt` include microseconds
   * `-T` show the time spend in system calls
   * `-yy` show ip:port pairs associated with socket file descripters
   * `-s 256` line length, default is 32
   * `-e trace=` can be used multiple times, `all` can be replaced by system calls. `-e mmap,munmap` eg.
   * `-c` displays a summary

`-e fault=syscall` makes `syscall` receive a generic error

`-e inject=syscall:error=error-type` or `-e inject=syscall:retval=return-val` specifies the error type or return value

For example:

`strace -e mmap:error=EPERM`

`-e malloc+free-@libc.so*` traces call to the `malloc` and `free` but to omit those that are done by `libc` library

`-e opendir+readdir+closedir` traces only the `opendir`, `readdir`, and `closedir` calls

Tracing a running program, add `-p${ps -C program -o pid h)` to the end of command

`ltrace -f -l library -e function ./program`

   * `-f` trace child processes or threads

Add a `-ppid` to the last command line to attach to a running program

`(gdb) catch syscall syscall-name` can stop GDB at calling `syscall-name`

`(gdb) catch signal signal-type` stops GDB at recieving `signal-type`

### Enable core dump

In `/etc/systemd/system.conf`

```
DumpCore=yes
DefaultLimitCORE=infinity
```

Apply changes

`#systemctl daemon-reexec`

Remove limits for core dump sizes

`#unlimit -c unlimited`

Collect additional information

`#sosreport`

### Analyze core dump file

Identify the executable file where the crash occurred

`en-unstrip -n --core=core-file` gets the buildid and filename

`eu-readelf -n executable-file` to confirm the build-id

`gdb -e executable-file -c core-file`

if the application's debugging information is available as a file

`(gdb) symbol-file program.debug`

### Dumping process memory

`gcore -o filename pid` or in GDB `(gdb) gcore core-file`
