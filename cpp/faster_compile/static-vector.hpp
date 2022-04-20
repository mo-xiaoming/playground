#include <array>
#include <stdexcept>

[[noreturn]] void throw_logic_error() {
  throw std::logic_error("overflew static vector");
}

class Static_vector {
  std::array<int, 10> arr{};
  std::size_t idx = 0;

public:
  constexpr void push_back(int i) {
    if (idx >= 10) {
      throw_logic_error();
    }
    arr[idx++] = i;
  }
  constexpr int size() const { return idx; }
};
