#include <benchmark/benchmark.h>

static void bench(benchmark::State& state) {
  auto a = new int[64 * 1024 * 1024]{};
  for (auto _ : state) {
    for (auto i = 0; i < 64 * 1024 * 1024; ++i) {
      a[i] *= 3;
    }
  }
}
BENCHMARK(bench);

static void bench_step(benchmark::State& state) {
  auto a = new int[64 * 1024 * 1024]{};
  for (auto _ : state) {
    for (auto i = 0; i < 64 * 1024 * 1024; i += 16) {
      a[i] *= 3;
    }
  }
}
BENCHMARK(bench_step);

BENCHMARK_MAIN();
