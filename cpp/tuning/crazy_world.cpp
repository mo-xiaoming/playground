#include <limits>
#include <vector>

#include <benchmark/benchmark.h>

#include "mx.hpp"

static void bench_cols_vary(benchmark::State& state) {
  auto const cols = state.range(0);

  constexpr auto const size = 128 * 1024;
  auto v = std::vector<char>{};
  v.reserve(size);
  std::generate_n(back_inserter(v), size,
                  [rg = mx::RandomGen<std::numeric_limits<int>::min(), std::numeric_limits<int>::max()>{}]() mutable {
                    return rg.next();
                  });
  for (auto _ : state) {
    auto sum = 0;
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < size / cols; ++row) {
        sum += v[row * cols + col];
      }
      benchmark::DoNotOptimize(sum);
    }
  }
}
BENCHMARK(bench_cols_vary)->RangeMultiplier(2)->Range(2, 64 * 1024);

BENCHMARK_MAIN();
