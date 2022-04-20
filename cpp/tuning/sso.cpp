#include <iostream>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "mx.hpp"

static void bench_col(benchmark::State& state) {
  auto v = std::vector<std::string>{};
  v.reserve(10'000);

  auto const size = state.range(0);
  for (auto _ : state) {
    std::fill_n(back_inserter(v), 10'000, std::string(size, 'X'));
  }
}
BENCHMARK(bench_col)->DenseRange(0, 32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
