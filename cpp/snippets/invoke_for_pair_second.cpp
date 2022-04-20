#include <functional>
#include <array>
#include <iostream>

template <typename Range, typename Callable>
void transform_print(Range const& r, Callable c) {
    for (auto const& e : r) {
        std::cout << std::invoke(c, e) << std::endl;
    }
}

int main() {
    std::array<std::pair<int, int>, 3> v{{{4, 40}, {5, 50}, {6, 60}}};
    transform_print(v, [](auto const& p) { return p.first * p.first; });
    transform_print(v, &std::pair<int, int>::second);
}
