| Name       | Description                                                                                                                                                                              |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Memcheck   | Detects memroy management problems by intercepting system calls and checking all read and write operations                                                                               |
| Cachegrind | Identifies the sources of cache misses by simulating the level 1 instruction cache (L1), level 1 data cache (D1), and unified level 2 cache (L2)                                         |
| Callgrind  | Generates a call graph representing the function call history                                                                                                                            |
| Helgrind   | Detects synchronmization errors in multithreaded C, C++, and Fortran programs that use **POSIX** threading primitives                                                                    |
| DRD        | Detects errors in multithreaded C and C++ programs that use **POSIX** threading primitives or any other threading concepts that are build on top of these **POSIX** threading primitives |
| Massif     | Monitors heap and stack usage                                                                                                                                                            |

`--tool=memcheck --leak-check=summery` is the default argument

`--gen-suppresions=yes` prints out a suppression for each reported error, for C++ names, use `--demangle=no` to get the mangled names

`-fno-inline` for C++ code, or use `--read-inline-info=yes` in Valgrind
