#include <vector>

#include <benchmark/benchmark.h>

static void bench(benchmark::State& state) {
  auto constexpr const lines = 1'000'003;

  auto v = std::vector<char>(lines * 64, 1);

  auto const skip = state.range(0);

  for (auto _ : state) {
    auto sum = 0;
    auto index = 0;
    for (auto i = 0; i < lines; ++i) {
      // sum += v[64 * ((skip * i) % lines)];
      sum += v[64 * index];
      index += skip;
      if (index >= lines) {
        index -= lines;
      }
    }
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(bench)->DenseRange(1, 100)->DisplayAggregatesOnly(true);

BENCHMARK_MAIN();
