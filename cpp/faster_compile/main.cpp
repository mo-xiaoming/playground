#include "static-vector.hpp"

int main() {
  Static_vector v;
  for (int i = 0; i < 20; ++i) {
    v.push_back(i);
  }
}
