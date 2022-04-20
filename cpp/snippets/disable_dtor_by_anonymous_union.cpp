#include <iostream>

struct Bar {
  ~Bar() { std::cout << "~Bar\n"; }
};

struct Foo {
  ~Foo() { std::cout << "~Foo\n"; }
  void destroy_member() { destroy_early.~Bar(); }

private:
  union {
    Bar destroy_early;
  };
};

int main() {
  Foo my_foo{};
  my_foo.destroy_member();
  std::cout << "done destroying member\n";
} // my_foo destroyed, but `destroy_early` not double-destroyed
