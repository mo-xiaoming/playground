#include <algorithm>
#include <array>
#include <iterator>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

constexpr auto SZ = 10000;
constexpr char const* V = "verrrrrrrrrrrrrrrrrrrry looooooooooooooooooooong";

static void copy(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto v0 = std::vector<std::string>{};
        v0.reserve(SZ);
        std::fill_n(back_inserter(v0), SZ, V);
        state.ResumeTiming();
        auto v1 = std::vector<std::string>(cbegin(v0), cend(v0));
        benchmark::DoNotOptimize(v1.data());
    }
}
BENCHMARK(copy);

static void move(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto v0 = std::vector<std::string>{};
        v0.reserve(SZ);
        std::fill_n(back_inserter(v0), SZ, V);
        state.ResumeTiming();
        auto v1 = std::vector<std::string>(std::make_move_iterator(begin(v0)),
                                           std::make_move_iterator(end(v0)));
        benchmark::DoNotOptimize(v1.data());
    }
}
BENCHMARK(move);

BENCHMARK_MAIN();
