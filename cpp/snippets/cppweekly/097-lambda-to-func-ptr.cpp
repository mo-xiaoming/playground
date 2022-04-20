#include <vector>
#include <functional>

int main() {
    // much more efficient than std::vector<std::function<bool(int, int)>> v;
    std::vector<bool (*)(int, int)> v;
    v.emplace_back([](int, int) { return false; });
}
