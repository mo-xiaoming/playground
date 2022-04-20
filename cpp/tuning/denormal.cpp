#include <cstdlib>
#include <vector>

namespace {
auto v = std::vector<float>(10 * 1024 * 1024, 1.0F);
}

auto main(int argc, char** argv) -> int {
    auto const f = static_cast<float>(std::atof(argv[1]));
    for (auto& i : v) {
        i *= f;
    }
    return v.back();
}
