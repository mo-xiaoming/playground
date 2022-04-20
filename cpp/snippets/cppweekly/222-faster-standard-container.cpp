#include <vector>
#include <memory_resource>
#include <iomanip>
#include <iostream>

int main() {
    std::byte buf[1024];
    // not suitable for gradually growed memory
    // because of vector reallocated memory on growth, not reuse
    auto mr = std::pmr::monotonic_buffer_resource(buf, sizeof buf);
    auto v1 = std::pmr::vector<int>({1, 2, 3}, &mr);
    auto v2 = std::pmr::vector<int>({4, 5, 6}, &mr);
    for (auto i = 0U; i < sizeof buf; ++i) {
        std::cout << std::hex << (unsigned int)buf[i] << (i%16==15?'\n':' ');
    }
}
