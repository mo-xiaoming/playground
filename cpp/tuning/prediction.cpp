#include <algorithm>
#include <vector>

#include <benchmark/benchmark.h>

#include "mx.hpp"

namespace {
auto rg = mx::RandomGen<0, 10>{};

auto fill_vector(std::size_t sz) -> std::vector<int> {
    auto v = std::vector<int>{};
    v.reserve(sz);
    std::generate_n(back_inserter(v), sz, [] { return rg.next(); });
    return v;
}

auto main_loop(benchmark::State& state, std::vector<int> const& v) {
    auto sum = 0;
    for (auto _ : state) {
        for (auto i : v) {
            if (i < 6) {
                sum += i;
            }
        }
    }
    benchmark::DoNotOptimize(&sum);
}

} // namespace

static void sorted(benchmark::State& state) {
    auto v = fill_vector(state.range(0));
    std::sort(begin(v), end(v));
    main_loop(state, v);
}

static void nosort(benchmark::State& state) {
    auto v = fill_vector(state.range(0));
    main_loop(state, v);
}

#define BENCH(NAME) BENCHMARK(NAME)->Range(10, 1'000'000)

BENCH(sorted);
BENCH(nosort);

BENCHMARK_MAIN();
