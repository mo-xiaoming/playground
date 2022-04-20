#include <cmath>

#include <benchmark/benchmark.h>

static void bench_1_4(benchmark::State& state) {
  for (auto _ : state) {
    auto a = std::pow(1.000'000'000'000'01, 1.4);
    benchmark::DoNotOptimize(&a);
  }
}
BENCHMARK(bench_1_4);

static void bench_1_5(benchmark::State& state) {
  for (auto _ : state) {
    auto a = std::pow(1.000'000'000'000'01, 1.5);
    benchmark::DoNotOptimize(&a);
  }
}
BENCHMARK(bench_1_5);

BENCHMARK_MAIN();
