#include <cstring>
#include <thread>
#include <vector>

#define REPETITION 1024 * 1024 * 1024UL
#define SIZE 512

void thread_fn(int tid, double* data, double* end) {
    for (std::size_t i = 0U; i < REPETITION; ++i) {
        for (double* ptr = data; ptr < end; ++ptr) {
            ptr[tid * 16] += i;
        }
    }
}

auto main(int argc, char** argv) -> int {
    auto thread_count = static_cast<std::size_t>(std::stoi(argv[1]));
    auto items = std::vector<double>(SIZE, 0);
    auto threads = std::vector<std::thread>{};
    for (auto i = 0; i < thread_count; ++i) {
        threads.emplace_back(thread_fn, i, items.data(), items.data() + 1);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}
