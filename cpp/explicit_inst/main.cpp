#include "arg.cpp"
#include "mock_arg.hpp"

template <typename T> void call_sut() {
  auto sut = Sut<T>();
  sut.foo();
  sut.t();
}

int main() {
  call_sut<Arg>();
  call_sut<Mock_arg>();
}

// https://stackoverflow.com/questions/2351148/explicit-template-instantiation-when-is-it-used
