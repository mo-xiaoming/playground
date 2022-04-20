#pragma once

#include "arg.hpp"

template <typename T = Arg> struct Sut {
  void foo();
  T &t();

private:
  T t_;
};
