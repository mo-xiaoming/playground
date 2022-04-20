#include <initializer_list>
#include <iostream>

template <typename ... T>
void print2(T const& ... t) {
    (void)std::initializer_list<int>{([](auto const &t) { std::cout << t << ' '; }(t), 0)...};
}

template <typename ... T>
void print0(T const& ... t) {
    // initializer_list and comma operator both have well defined evaluation order, from left to right
    (void)std::initializer_list<int>{(std::cout << t << ' ', 0)...};
}

template <typename ... T>
void print1(T const& ... t) {
    ((std::cout << t << ' '), ...);
}

int main() {
    print1(1, "hello", '3', 7.6);
}
