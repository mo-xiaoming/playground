#include <iostream>

// C++11, variadic templates

template <int...> struct sum; // declare primary templates

template <> struct sum<> {
  static constexpr int value = 0;
}; // full specialization for zero arguments

template <int i, int... tail> struct sum<i, tail...> {
  static constexpr int value = i + sum<tail...>::value;
}; // partial specialization for at least one argument

template <int...> struct product;

template <> struct product<> { static constexpr int value = 1; };

template <int i, int... tail> struct product<i, tail...> {
  static constexpr int value = i * product<tail...>::value;
};

// C++17, fold expressions

template <typename... Args> auto sum1(Args const &... args) {
  return (args + ...);
}

template <typename... Args> auto product1(Args const &... args) {
  return (args * ...);
}

int main() {
  std::cout << "sum<1, 2, 3, 4, 5>::value: " << sum<1, 2, 3, 4, 5>::value
            << std::endl;
  std::cout << "product<1, 2, 3, 4, 5>::value: "
            << product<1, 2, 3, 4, 5>::value << std::endl;

  std::cout << "sum1(1, 2, 3, 4, 5): " << sum1(1, 2, 3, 4, 5) << std::endl;
  std::cout << "product1(1, 2, 3, 4, 5): " << product1(1, 2, 3, 4, 5)
            << std::endl;
}
