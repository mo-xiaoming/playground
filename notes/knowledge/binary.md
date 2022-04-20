`ldd a.out` libraries that link against
    - `readelf -d a.out |grep NEEDED`
	- `objdump -x a.out |grep NEEDED`
	- `objdump -p a.out |grep NEEDED`


`nm -gC a.a` dump external symbols in .a

`objdump -s --section .comment a.out` compiler info

`objdump -TC a.out` external functions that needed

`size a.out` list section sizes

`readelf -x .rodata a.out` show static strings

`readelf -h a.out` show header

`readelf -s a.out` show symbols

`xxd` read write hex/binary files

`xxd -l4 -i /bin/ls` read 4 bytes in C array style

