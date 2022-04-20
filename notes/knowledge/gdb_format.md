#### gdb how-to

```bash
objdump -h a.out
```

```bash
readelf -S a.out
```

Debug sections are usually in DWARF format. For ELF binaries these debug sections have names like `.debug_*`, e.g.`.debug_info` or `.debug_loc`.


Debug info a collection of DIEs(Debug Info Entries). Each DIE has a tag specifying what kind of DIE it is and attributes that describes this DIE - things like variable name and line number.

```bash
objdump -g a.out
```

Under `.debug_info` section, find all DIEs with tag `DW_TAG_compile_unit`. The DIE has 2 main attributes `DW_AT_comp_dir`(compilation directory) and `DW_AT_name` - path to the source file.

Contents of the .debug_info section:

>  Compilation Unit @ offset 0x0:
>   Length:        0x222d (32-bit)
>   Version:       4
>   Abbrev Offset: 0x0
>   Pointer Size:  8
> <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
>    <c>   DW_AT_producer    : (indirect string, offset: 0xb6b): GNU C99 6.3.1 20161221 (Red Hat 6.3.1-1) -mtune=generic -march=x86-64 -g -Og -std=c99 
>    <10>   DW_AT_language    : 12   (ANSI C99)
>    <11>   DW_AT_name        : (indirect string, offset: 0x10ec): ./Programs/python.c
>    <15>   DW_AT_comp_dir    : (indirect string, offset: 0x7a): /home/avd/dev/cpython
>    <19>   DW_AT_low_pc      : 0x41d2f6
>    <21>   DW_AT_high_pc     : 0x1b3 
>    <29>   DW_AT_stmt_list   : 0x0   

It reads like, for address range from `DW_AT_low_pc` = `0x41d2f6` to `DW_AT_low_pc + DW_AT_high_pc` = `0x41d2f6` + `0x1b3` = `0x41d4a9` source code file is the `./Programs/python.c` located in `/home/avd/dev/cpython`
