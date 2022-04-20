#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <unistd.h>

using namespace std::literals::chrono_literals;

namespace {
void* my_malloc(size_t size) noexcept {
    auto previous_break = sbrk(size);
    return previous_break == (void*)-1 ? nullptr : previous_break;
}
} // namespace

auto main(int argc, char** argv) -> int {
    auto a = std::vector<char*>(128 * 50000);
    auto m = std::malloc;
    if (argc > 1) {
        std::cout << "change to my_alloc\n";
        m = my_malloc;
    }
    for (auto& i : a) {
        i = (char*)m(8);
    }
    return *a.back();
}
