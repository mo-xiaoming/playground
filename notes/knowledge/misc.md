export PS1="\[\e[35m\]\u\[\e[m\]@\[\e[33m\]\h\[\e[m\] \t \W \[\e[34m\]\$\[\e[m\] "

# build LLVM with limited memory
sudo apt install lld clang
ninja -j1
[MemorySanitizerLibcxxHowTo](https://github.com/google/sanitizers/wiki/MemorySanitizerLibcxxHowTo)
cmake -GNinja -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld -DLLVM_USE_SANITIZER=MemoryWithOrigins -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/llvm-msan-build ../llvm
cmake --build . -- cxx cxxabi

MSAN_CFLAGS="-fsanitize=memory -stdlib=libc++ -L/path/to/libcxx_msan/lib -lc++abi -I/path/to/libcxx_msan/include -I/path/to/libcxx_msan/include/c++/v1"
cmake ../googletest -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS="$MSAN_CFLAGS" -DCMAKE_CXX_FLAGS="$MSAN_CFLAGS"
make -j12

#delete empty lines
sed 's/^[[:space:]]*$/d'

#list files process opened
ls -l /proc/{pid}/fd/
lsof -p {pid}

#port range
cat /proc/sys/net/ipv4/ip_local_port_range

# run command on cpu #1 and #2
taskset -c 1,2 cmd arg1 arg2

# find largest files
find . -name .wdmc -prune -o -type f -exec stat -f "%z %N" "{}" \; |sort -n

#list all ports
lsof -i

#which process using port
lsof -i :[PORT]

# find out top biggest directories under some folder
du -a /home | sort -nr | head -n1

du -a: display all files and folders

sort -n: compare according to string numerical value
     -r: reverse

du -hs * |sort -rh |head -n1

du -Sh |sort -rh |head -n1

du -h: human readable format
   -S: do not include size of subdirectories
   -s: display only a total for each argument

sort -h: compare human readable numbers

# find ip
dig 2daygeek.com

host 2daygeek.com

nsloopup -q=A 2daygeek.com
