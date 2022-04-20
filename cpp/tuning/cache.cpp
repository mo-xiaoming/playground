#include <algorithm>
#include <iostream>

#include <benchmark/benchmark.h>
#include <limits>

#include "mx.hpp"

static void bench_col(benchmark::State& state) {
  auto const bytes = 1U << static_cast<std::make_unsigned_t<decltype(state.range(0))>>(state.range(0));
  auto const count = bytes / sizeof(int) / 2;
  auto v = std::vector<int>{};
  v.reserve(count);
  std::generate_n(back_inserter(v), count,
                  [rg = mx::RandomGen<std::numeric_limits<int>::min(), std::numeric_limits<int>::max()>{}]() mutable {
                    return rg.next();
                  });
  auto indices = std::vector<int>{};
  indices.reserve(count);
  std::generate_n(back_inserter(indices), count,
                  [rg = mx::RandomGen1<decltype(count - 1)>{0, count - 1}]() mutable { return rg.next(); });

  for (auto _ : state) {
    auto sum = 0L;
    for (auto i : indices) {
      sum += v[i];
    }
    benchmark::DoNotOptimize(&sum);
  }
  state.SetBytesProcessed(bytes * state.iterations());
  state.SetLabel(std::to_string(bytes / 1024) + "kb");
}
BENCHMARK(bench_col)->DenseRange(13, 26)->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
