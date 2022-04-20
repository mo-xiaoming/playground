#include <iostream>

template <typename ... T>
auto div1(T ...t) {
    return (t / ...); //  1. / (2. / 3.) == 1.5
}

template <typename ... T>
auto div2(T ...t) {
    return (... / t); // (1. / 2.) / 3. == 0.167
}

template <typename ... T>
auto avg(T ...t) {
    auto const start = 0;
    return (t + ... + start) / sizeof...(t);
}

int main() {
    std::cout << div1(1., 2., 3.) << '\n';
    std::cout << div2(1., 2., 3.) << '\n';
    std::cout << avg(1., 2., 3.) << '\n';
}
