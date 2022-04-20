#include <limits>
#include <vector>

#include <benchmark/benchmark.h>

#include "mx.hpp"

static void bench_cols_vary(benchmark::State& state) {
    auto const step = state.range(0);

    constexpr auto const size = 40 * 1024 * 1024;
    auto v = std::vector<char>{};
    v.reserve(size);
    std::generate_n(
        back_inserter(v), size,
        [rg = mx::RandomGen<std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::max()>{}]() mutable {
            return rg.next();
        });
    auto p = std::vector<char*>{};
    auto total = 20480;
    p.reserve(total);
    for (auto i = 0; total > 0; --total, i += step) {
        p.push_back(&v[i]);
    }
    for (auto _ : state) {
        for (auto i : p) {
            *i += 1;
        }
    }
}
BENCHMARK(bench_cols_vary)
    ->Arg(64)
    ->Arg(128)
    ->Arg(480)
    ->Arg(512)
    ->Arg(542)
    ->Arg(960)
    ->Arg(1024)
    ->Arg(1124)
    ->Arg(2048);

BENCHMARK_MAIN();
