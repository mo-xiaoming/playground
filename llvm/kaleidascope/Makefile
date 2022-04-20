all: toy toy2

toy: toy.cpp
	clang++ -fno-limit-debug-info -fsanitize=address,undefined -ggdb3 -O0 $^ `llvm-config --cxxflags` -o $@

toy2: toy2.cpp
	clang++ -fno-limit-debug-info -fsanitize=address,undefined -ggdb3 -O0 $^ `llvm-config --cxxflags --ldflags --system-libs --libs core` -o $@
