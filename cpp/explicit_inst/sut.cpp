#include "sut.hpp"

template <typename T> void Sut<T>::foo() { t_.foo(); }

template <typename T> T &Sut<T>::t() { return t_; }
