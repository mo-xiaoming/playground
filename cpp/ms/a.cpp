#include <algorithm>
#include <cassert>
#include <vector>
#include <array>

std::vector<char> merge(std::vector<char> const& a, std::vector<char> const& b, std::vector<char> const& c) {
    auto result = std::vector<char>();
    result.reserve(a.size() + b.size() + c.size());
    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    result.insert(result.end(), c.begin(), c.end());
    std::sort(result.begin(), result.end());
    return result;
}

int main() {
    auto const a0 = std::vector<char>{'a', 'b', 'c'};
    auto const a1 = std::vector<char>{'b', 'a', 'c'};
    auto const a2 = std::vector<char>{'c', 'b', 'a'};
    auto const c = merge(a0, a1, a2);
    assert(std::equal(std::cbegin(c), std::cend(c), std::cbegin(std::array<char, 9>{'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'})));
}
