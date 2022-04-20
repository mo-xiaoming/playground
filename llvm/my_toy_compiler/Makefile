all: parser

OBJS = parser.o  \
       codegen.o \
       main.o    \
       tokens.o  \
       corefn.o  \
	   native.o  \

LLVMCONFIG = llvm-config
CPPFLAGS = -isystem `$(LLVMCONFIG) --cppflags` -Wall -Wextra -ggdb3 -O0 -fno-omit-frame-pointer# -fsanitize=address,undefined
LDFLAGS = `$(LLVMCONFIG) --ldflags` -lpthread -ldl -lz -lncurses -rdynamic# -fsanitize=address,undefined
LIBS = `$(LLVMCONFIG) --libs`

clean:
	$(RM) -rf parser.cpp parser.hpp parser tokens.cpp $(OBJS)

test: parser example.txt
	cat example.txt | ./parser

parser: $(OBJS)
	g++ -o $@ $(OBJS) $(LIBS) $(LDFLAGS)

%.o: %.cpp %.h
	g++ -c $(CPPFLAGS) -o $@ $<

tokens.cpp: tokens.l parser.hpp
	flex -o $@ $^

parser.hpp: parser.cpp

parser.cpp: parser.y
	bison -d -o $@ $^
