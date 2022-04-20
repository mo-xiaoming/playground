#include <benchmark/benchmark.h>

static void bench(benchmark::State& state) {
  auto a = new int[4 * 1024 * 1024]{};
  auto const step = state.range(0);
  for (auto _ : state) {
    for (auto i = 0; i < 4 * 1024 * 1024; i += step) {
      a[i] *= 3;
    }
  }
}
BENCHMARK(bench)->RangeMultiplier(2)->Range(2, 1024);

BENCHMARK_MAIN();
