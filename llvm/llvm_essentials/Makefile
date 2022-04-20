toy : toy.cpp
	clang++ -Og -ggdb3 -fno-omit-frame-pointer -fsanitize=address,undefined $^ `/usr//local/Cellar/llvm/12.0.0_1/bin/llvm-config --cxxflags --ldflags --system-libs --libs core` -o $@

toy1 : toy1.cpp
	clang++ -Og -ggdb3 -fno-omit-frame-pointer $^ `llvm-config --cxxflags --ldflags --system-libs --libs core` -o $@

toy2 : toy2.cpp
	clang++ -Og -ggdb3 -fno-omit-frame-pointer $^ `llvm-config --cxxflags --ldflags --system-libs --libs core` -o $@

toy3 : toy3.cpp
	clang++ -Og -ggdb3 -fno-omit-frame-pointer $^ `llvm-config --cxxflags --ldflags --system-libs --libs core` -o $@

toy4 : toy4.cpp
	clang++ -Og -ggdb3 -fno-omit-frame-pointer $^ `llvm-config --cxxflags --ldflags --system-libs --libs core` -o $@

toy5 : toy5.cpp
	clang++ -Og -ggdb3 -fno-omit-frame-pointer $^ `llvm-config --cxxflags --ldflags --system-libs --libs core` -o $@

clean:
	@rm -f *.o toy toy1 toy2 toy3 toy4 toy5

